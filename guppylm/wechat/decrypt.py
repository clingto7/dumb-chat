"""WeChat database decryption — extract keys from process memory and decrypt SQLCipher 4 databases.

Ported from wechat-decrypt (https://github.com/ylytdeng/wechat-decrypt):
- find_all_keys_linux.py — Linux /proc/<pid>/mem scanning
- key_scan_common.py — HMAC key verification
- decrypt_db.py — AES-256-CBC page-level decryption
"""

import functools
import hashlib
import hmac as hmac_mod
import json
import os
import re
import struct
import sys
import time

from .config import WechatDecryptConfig

print = functools.partial(print, flush=True)

# SQLCipher 4 constants
PAGE_SZ = 4096
KEY_SZ = 32
SALT_SZ = 16
IV_SZ = 16
RESERVE_SZ = 80  # 16-byte IV + 64-byte HMAC
SQLITE_HDR = b"SQLite format 3\x00"


# ── HMAC key verification ────────────────────────────────────────────────────


def verify_enc_key(enc_key: bytes, db_page1: bytes) -> bool:
    """Verify an encryption key against page 1 of a database using HMAC-SHA512."""
    salt = db_page1[:SALT_SZ]
    mac_salt = bytes(b ^ 0x3A for b in salt)
    mac_key = hashlib.pbkdf2_hmac("sha512", enc_key, mac_salt, 2, dklen=KEY_SZ)
    hmac_data = db_page1[SALT_SZ : PAGE_SZ - RESERVE_SZ + IV_SZ]
    stored_hmac = db_page1[PAGE_SZ - 64 : PAGE_SZ]
    hm = hmac_mod.new(mac_key, hmac_data, hashlib.sha512)
    hm.update(struct.pack("<I", 1))
    return hm.digest() == stored_hmac


# ── Database file collection ─────────────────────────────────────────────────


def collect_db_files(db_dir: str):
    """Walk db_dir and collect all .db files with their salts and page 1 data.

    Returns:
        db_files: list of (rel_path, abs_path, size, salt_hex, page1_bytes)
        salt_to_dbs: dict mapping salt_hex -> list of rel_paths
    """
    db_files = []
    salt_to_dbs = {}
    for root, dirs, files in os.walk(db_dir):
        for name in files:
            if not name.endswith(".db") or name.endswith(("-wal", "-shm")):
                continue
            path = os.path.join(root, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size < PAGE_SZ:
                continue
            with open(path, "rb") as f:
                page1 = f.read(PAGE_SZ)
            rel = os.path.relpath(path, db_dir)
            salt = page1[:SALT_SZ].hex()
            db_files.append((rel, path, size, salt, page1))
            salt_to_dbs.setdefault(salt, []).append(rel)
    return db_files, salt_to_dbs


# ── Memory scanning for keys ────────────────────────────────────────────────


def scan_memory_for_keys(data, hex_re, db_files, salt_to_dbs, key_map,
                         remaining_salts, base_addr, pid):
    """Scan a chunk of memory data, match hex patterns and verify keys.

    Returns the number of hex patterns found in this chunk.
    """
    matches = 0
    for m in hex_re.finditer(data):
        hex_str = m.group(1).decode()
        addr = base_addr + m.start()
        matches += 1
        hex_len = len(hex_str)

        if hex_len == 96:
            # Standard pattern: 64-hex enc_key + 32-hex salt
            enc_key_hex = hex_str[:64]
            salt_hex = hex_str[64:]
            if salt_hex in remaining_salts:
                enc_key = bytes.fromhex(enc_key_hex)
                for rel, path, sz, s, page1 in db_files:
                    if s == salt_hex and verify_enc_key(enc_key, page1):
                        key_map[salt_hex] = enc_key_hex
                        remaining_salts.discard(salt_hex)
                        dbs = salt_to_dbs[salt_hex]
                        print(f"\n  [FOUND] salt={salt_hex}")
                        print(f"    enc_key={enc_key_hex}")
                        print(f"    PID={pid} addr: 0x{addr:016X}")
                        print(f"    databases: {', '.join(dbs)}")
                        break

        elif hex_len == 64:
            # Key only (no salt appended)
            if not remaining_salts:
                continue
            enc_key_hex = hex_str
            enc_key = bytes.fromhex(enc_key_hex)
            for rel, path, sz, salt_hex_db, page1 in db_files:
                if salt_hex_db in remaining_salts and verify_enc_key(enc_key, page1):
                    key_map[salt_hex_db] = enc_key_hex
                    remaining_salts.discard(salt_hex_db)
                    dbs = salt_to_dbs[salt_hex_db]
                    print(f"\n  [FOUND] salt={salt_hex_db}")
                    print(f"    enc_key={enc_key_hex}")
                    print(f"    PID={pid} addr: 0x{addr:016X}")
                    print(f"    databases: {', '.join(dbs)}")
                    break

        elif hex_len > 96 and hex_len % 2 == 0:
            # Longer pattern: enc_key in first 64 hex, salt in last 32 hex
            enc_key_hex = hex_str[:64]
            salt_hex = hex_str[-32:]
            if salt_hex in remaining_salts:
                enc_key = bytes.fromhex(enc_key_hex)
                for rel, path, sz, s, page1 in db_files:
                    if s == salt_hex and verify_enc_key(enc_key, page1):
                        key_map[salt_hex] = enc_key_hex
                        remaining_salts.discard(salt_hex)
                        dbs = salt_to_dbs[salt_hex]
                        print(f"\n  [FOUND] salt={salt_hex} (long hex {hex_len})")
                        print(f"    enc_key={enc_key_hex}")
                        print(f"    PID={pid} addr: 0x{addr:016X}")
                        print(f"    databases: {', '.join(dbs)}")
                        break

    return matches


def cross_verify_keys(db_files, salt_to_dbs, key_map):
    """Try to verify unmatched salts using keys found for other salts."""
    missing_salts = set(salt_to_dbs.keys()) - set(key_map.keys())
    if not missing_salts or not key_map:
        return
    print(f"\n{len(missing_salts)} salts unmatched, trying cross-verification...")
    for salt_hex in list(missing_salts):
        for rel, path, sz, s, page1 in db_files:
            if s == salt_hex:
                for known_salt, known_key_hex in key_map.items():
                    enc_key = bytes.fromhex(known_key_hex)
                    if verify_enc_key(enc_key, page1):
                        key_map[salt_hex] = known_key_hex
                        print(f"  [CROSS] salt={salt_hex} matches key from salt={known_salt}")
                        missing_salts.discard(salt_hex)
                break


def save_results(db_files, salt_to_dbs, key_map, db_dir, out_file):
    """Save key extraction results to JSON."""
    print(f"\n{'=' * 60}")
    print(f"Result: {len(key_map)}/{len(salt_to_dbs)} salts matched")

    result = {}
    for rel, path, sz, salt_hex, page1 in db_files:
        if salt_hex in key_map:
            result[rel] = {
                "enc_key": key_map[salt_hex],
                "salt": salt_hex,
                "size_mb": round(sz / 1024 / 1024, 1),
            }
            print(f"  OK: {rel} ({sz / 1024 / 1024:.1f}MB)")
        else:
            print(f"  MISSING: {rel} (salt={salt_hex})")

    if not result:
        raise RuntimeError("No keys extracted from any WeChat process")

    result["_db_dir"] = db_dir
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    os.chmod(out_file, 0o600)
    print(f"\nKeys saved to: {out_file}")

    missing = [rel for rel, path, sz, salt_hex, page1 in db_files if salt_hex not in key_map]
    if missing:
        print(f"\nDatabases without keys:")
        for rel in missing:
            print(f"  {rel}")


# ── Process discovery ────────────────────────────────────────────────────────


def _safe_readlink(path):
    try:
        return os.path.realpath(os.readlink(path))
    except OSError:
        return ""


_KNOWN_COMMS = {"wechat", "wechatappex", "weixin"}
_INTERPRETER_PREFIXES = ("python", "bash", "sh", "zsh", "node", "perl", "ruby")


def _is_wechat_process(pid):
    """Check if pid is a WeChat process."""
    if pid == os.getpid():
        return False
    try:
        with open(f"/proc/{pid}/comm") as f:
            comm = f.read().strip()
        if comm.lower() in _KNOWN_COMMS:
            return True
        exe_path = _safe_readlink(f"/proc/{pid}/exe")
        exe_name = os.path.basename(exe_path)
        if any(exe_name.lower().startswith(p) for p in _INTERPRETER_PREFIXES):
            return False
        return "wechat" in exe_name.lower() or "weixin" in exe_name.lower()
    except (PermissionError, FileNotFoundError, ProcessLookupError):
        return False


def find_wechat_pid() -> int | None:
    """Find the running WeChat process PID on Linux.

    Returns the PID of the WeChat process with the largest RSS, or None.
    """
    pids = []
    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)
        try:
            if not _is_wechat_process(pid):
                continue
            with open(f"/proc/{pid}/statm") as f:
                rss_pages = int(f.read().split()[1])
            rss_kb = rss_pages * 4
            pids.append((pid, rss_kb))
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue

    if not pids:
        return None

    pids.sort(key=lambda item: item[1], reverse=True)
    return pids[0][0]


def _get_pids():
    """Return all suspected WeChat main process (pid, rss_kb) pairs, sorted by RSS descending."""
    pids = []
    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)
        try:
            if not _is_wechat_process(pid):
                continue
            with open(f"/proc/{pid}/statm") as f:
                rss_pages = int(f.read().split()[1])
            rss_kb = rss_pages * 4
            pids.append((pid, rss_kb))
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue

    if not pids:
        raise RuntimeError("No WeChat process detected on Linux")

    pids.sort(key=lambda item: item[1], reverse=True)
    for pid, rss_kb in pids:
        exe_path = _safe_readlink(f"/proc/{pid}/exe")
        print(f"[+] WeChat PID={pid} ({rss_kb // 1024}MB) {exe_path}")
    return pids


_SKIP_MAPPINGS = {"[vdso]", "[vsyscall]", "[vvar]"}
_SKIP_PATH_PREFIXES = ("/usr/lib/", "/lib/", "/usr/share/")


def _get_readable_regions(pid):
    """Parse /proc/<pid>/maps and return readable memory regions."""
    regions = []
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            if "r" not in parts[1]:
                continue
            if len(parts) >= 6:
                mapping_name = parts[5]
                if mapping_name in _SKIP_MAPPINGS:
                    continue
                mapping_lower = mapping_name.lower()
                if (any(mapping_name.startswith(p) for p in _SKIP_PATH_PREFIXES)
                        and "wcdb" not in mapping_lower
                        and "wechat" not in mapping_lower
                        and "weixin" not in mapping_lower):
                    continue
            start_s, end_s = parts[0].split("-")
            start = int(start_s, 16)
            size = int(end_s, 16) - start
            if 0 < size < 500 * 1024 * 1024:
                regions.append((start, size))
    return regions


def _check_permissions():
    """Check if we have permission to read process memory (root or CAP_SYS_PTRACE)."""
    if os.geteuid() == 0:
        return
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("CapEff:"):
                    cap_eff = int(line.split(":")[1].strip(), 16)
                    CAP_SYS_PTRACE = 1 << 19
                    if cap_eff & CAP_SYS_PTRACE:
                        return
                    break
    except (OSError, ValueError):
        pass
    print("[!] Root or CAP_SYS_PTRACE required to read process memory")
    print("    Use: sudo python3 -m guppylm wechat-decrypt")
    print("    Or:  sudo setcap cap_sys_ptrace=ep $(which python3)")
    sys.exit(1)


# ── Key extraction ────────────────────────────────────────────────────────────


def extract_keys(wechat_data_dir: str, output_path: str = "wechat_data/all_keys.json") -> dict:
    """Extract encryption keys from running WeChat process memory on Linux.

    Scans /proc/<pid>/mem for the WCDB cached key pattern x'<64hex><32hex>',
    validates keys via HMAC check against page 1 of each database.

    Returns:
        dict mapping db_relative_path -> {"enc_key": hex, "salt": hex, ...}
    """
    _check_permissions()

    # Resolve wxid subdirectory
    db_dir = _resolve_db_dir(wechat_data_dir)

    print("=" * 60)
    print("  WeChat Database Key Extraction (Linux Memory Scan)")
    print("=" * 60)

    # 1. Collect DB files and salts
    db_files, salt_to_dbs = collect_db_files(db_dir)
    if not db_files:
        raise RuntimeError(f"No .db files found in {db_dir}")

    print(f"\nFound {len(db_files)} databases, {len(salt_to_dbs)} unique salts")
    for salt_hex, dbs in sorted(salt_to_dbs.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  salt {salt_hex}: {', '.join(dbs)}")

    # 2. Find WeChat process
    pids = _get_pids()

    hex_re = re.compile(rb"x'([0-9a-fA-F]{64,192})'")
    key_map = {}
    remaining_salts = set(salt_to_dbs.keys())
    all_hex_matches = 0
    t0 = time.time()

    for pid, rss_kb in pids:
        try:
            regions = _get_readable_regions(pid)
        except PermissionError:
            print(f"[WARN] Cannot read /proc/{pid}/maps, permission denied, skipping")
            continue
        except (FileNotFoundError, ProcessLookupError):
            print(f"[WARN] PID {pid} exited, skipping")
            continue

        total_bytes = sum(s for _, s in regions)
        total_mb = total_bytes / 1024 / 1024
        print(f"\n[*] Scanning PID={pid} ({total_mb:.0f}MB, {len(regions)} regions)")

        scanned_bytes = 0
        try:
            mem = open(f"/proc/{pid}/mem", "rb")
        except PermissionError:
            print(f"[WARN] Cannot open /proc/{pid}/mem, permission denied, skipping")
            continue
        except (FileNotFoundError, ProcessLookupError):
            print(f"[WARN] PID {pid} exited, skipping")
            continue

        # TOCTOU defense: re-check after opening mem
        if not _is_wechat_process(pid):
            print(f"[WARN] PID {pid} is no longer WeChat, skipping")
            mem.close()
            continue

        try:
            for reg_idx, (base, size) in enumerate(regions):
                try:
                    mem.seek(base)
                    data = mem.read(size)
                except (OSError, ValueError):
                    continue
                scanned_bytes += len(data)

                all_hex_matches += scan_memory_for_keys(
                    data, hex_re, db_files, salt_to_dbs,
                    key_map, remaining_salts, base, pid,
                )

                if (reg_idx + 1) % 200 == 0:
                    elapsed = time.time() - t0
                    progress = scanned_bytes / total_bytes * 100 if total_bytes else 100
                    print(
                        f"  [{progress:.1f}%] {len(key_map)}/{len(salt_to_dbs)} salts matched, "
                        f"{all_hex_matches} hex patterns, {elapsed:.1f}s"
                    )
        finally:
            mem.close()

        if not remaining_salts:
            print(f"\n[+] All keys found, skipping remaining processes")
            break

    elapsed = time.time() - t0
    print(f"\nScan complete: {elapsed:.1f}s, {len(pids)} processes, {all_hex_matches} hex patterns")

    # 3. Cross-verify and save
    cross_verify_keys(db_files, salt_to_dbs, key_map)
    save_results(db_files, salt_to_dbs, key_map, db_dir, output_path)

    # Return the saved keys
    with open(output_path) as f:
        return json.load(f)


# ── Database decryption ─────────────────────────────────────────────────────


def _derive_mac_key(enc_key: bytes, salt: bytes) -> bytes:
    """Derive the MAC key from the encryption key and salt."""
    mac_salt = bytes(b ^ 0x3A for b in salt)
    return hashlib.pbkdf2_hmac("sha512", enc_key, mac_salt, 2, dklen=KEY_SZ)


def decrypt_page(enc_key: bytes, page_data: bytes, pgno: int) -> bytes:
    """Decrypt a single 4096-byte page into a standard SQLite page."""
    from Crypto.Cipher import AES

    iv = page_data[PAGE_SZ - RESERVE_SZ : PAGE_SZ - RESERVE_SZ + IV_SZ]
    if pgno == 1:
        # Page 1: first 16 bytes are salt (not encrypted), data starts at offset 16
        encrypted = page_data[SALT_SZ : PAGE_SZ - RESERVE_SZ]
        cipher = AES.new(enc_key, AES.MODE_CBC, iv)
        decrypted = cipher.decrypt(encrypted)
        return bytes(bytearray(SQLITE_HDR + decrypted + b"\x00" * RESERVE_SZ))
    else:
        encrypted = page_data[: PAGE_SZ - RESERVE_SZ]
        cipher = AES.new(enc_key, AES.MODE_CBC, iv)
        decrypted = cipher.decrypt(encrypted)
        return decrypted + b"\x00" * RESERVE_SZ


def decrypt_database(db_path: str, key_hex: str, output_path: str) -> str:
    """Decrypt a single SQLCipher 4 encrypted database.

    Implements page-level AES-256-CBC decryption matching WCDB/SQLCipher 4 format.
    Does not use pysqlcipher3; operates on raw file bytes.

    Args:
        db_path: Path to the encrypted .db file.
        key_hex: 64-character hex string of the raw encryption key.
        output_path: Path to write the decrypted plain SQLite file.

    Returns:
        The path to the decrypted database.
    """
    enc_key = bytes.fromhex(key_hex)

    with open(db_path, "rb") as f:
        data = f.read()

    total_pages = len(data) // PAGE_SZ
    if total_pages < 1:
        raise ValueError(f"Database too small: {db_path}")

    # Extract salt from page 1
    page1 = data[:PAGE_SZ]
    salt = page1[:SALT_SZ]

    # Verify HMAC on page 1
    mac_key = _derive_mac_key(enc_key, salt)
    hmac_data = page1[SALT_SZ : PAGE_SZ - RESERVE_SZ + IV_SZ]
    stored_hmac = page1[PAGE_SZ - 64 : PAGE_SZ]
    hm = hmac_mod.new(mac_key, hmac_data, hashlib.sha512)
    hm.update(struct.pack("<I", 1))
    if hm.digest() != stored_hmac:
        raise ValueError(f"HMAC verification failed for {db_path}")

    # Decrypt all pages
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as out:
        for pgno in range(1, total_pages + 1):
            offset = (pgno - 1) * PAGE_SZ
            page_data = data[offset : offset + PAGE_SZ]
            decrypted = decrypt_page(enc_key, page_data, pgno)
            out.write(decrypted)

    # Validate by opening with sqlite3
    import sqlite3
    conn = sqlite3.connect(output_path)
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    conn.close()
    # Clean up residual -shm/-wal from validation
    for ext in ("-shm", "-wal"):
        p = output_path + ext
        if os.path.exists(p):
            os.remove(p)

    print(f"  Decrypted: {db_path} -> {output_path} ({len(tables)} tables)")
    return output_path


def decrypt_all(config: WechatDecryptConfig | None = None) -> list[str]:
    """Decrypt all WeChat databases found in the configured data directory.

    Steps:
    1. Extract keys from running WeChat process memory
    2. Decrypt each database for which a key was found
    3. Save decrypted databases to config.decrypted_dir

    Returns:
        List of paths to decrypted database files.
    """
    if config is None:
        config = WechatDecryptConfig()

    db_dir = _resolve_db_dir(config.wechat_data_dir)
    keys_path = config.keys_file
    decrypted_dir = config.decrypted_dir

    # 1. Extract keys (or load existing)
    if os.path.exists(keys_path):
        print(f"Loading existing keys from {keys_path}")
        with open(keys_path) as f:
            keys = json.load(f)
    else:
        print("No existing keys found, extracting from process memory...")
        keys = extract_keys(db_dir, keys_path)

    # 2. Decrypt databases
    os.makedirs(decrypted_dir, exist_ok=True)
    decrypted_paths = []

    for rel_path, info in keys.items():
        if rel_path.startswith("_"):
            continue  # Skip metadata keys like "_db_dir"
        enc_key_hex = info["enc_key"]
        src_path = os.path.join(db_dir, rel_path)
        dst_path = os.path.join(decrypted_dir, rel_path)

        if not os.path.exists(src_path):
            print(f"  SKIP: {src_path} not found")
            continue

        try:
            result_path = decrypt_database(src_path, enc_key_hex, dst_path)
            decrypted_paths.append(result_path)
        except Exception as e:
            print(f"  ERROR decrypting {rel_path}: {e}")

    print(f"\nDecrypted {len(decrypted_paths)} databases to {decrypted_dir}/")
    return decrypted_paths


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_db_dir(wechat_data_dir: str, wxid: str = "") -> str:
    """Resolve the db_storage directory path.

    If wxid is specified, uses it directly.
    Otherwise, auto-detects by looking for db_storage/ subdirectories.
    """
    base = wechat_data_dir
    if wxid:
        candidate = os.path.join(base, wxid, "db_storage")
        if os.path.isdir(candidate):
            return candidate
        raise FileNotFoundError(f"db_storage not found at {candidate}")

    # Auto-detect: find wxid_*/db_storage/
    candidates = []
    if os.path.isdir(base):
        for name in os.listdir(base):
            db_storage = os.path.join(base, name, "db_storage")
            if os.path.isdir(db_storage):
                candidates.append(db_storage)

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        print("Multiple WeChat user directories found:")
        for i, c in enumerate(candidates):
            print(f"  [{i}] {c}")
        choice = input("Select [0]: ").strip()
        idx = int(choice) if choice else 0
        return candidates[idx]
    else:
        raise FileNotFoundError(f"No db_storage/ found under {base}")
