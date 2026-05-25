"""WeChat chat record extraction from decrypted databases.

Reads plain SQLite databases produced by guppylm/wechat/decrypt.py and extracts:
- Contact list (from contact.db)
- Session list (from session.db)
- Chat messages (from message_0.db, biz_message_0.db)

Uses dynamic table/column discovery since WeChat 4.x Linux schema may vary.
"""

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .config import WechatExtractConfig


@dataclass
class WechatContact:
    wxid: str           # e.g. "wxid_abc123" or "12345678@chatroom"
    alias: str          # WeChat alias / username
    nickname: str       # Display name
    remark: str         # User's custom remark
    is_group: bool      # True if this is a group chat

    @property
    def display_name(self) -> str:
        """Best available display name: remark > nickname > alias > wxid."""
        return self.remark or self.nickname or self.alias or self.wxid


@dataclass
class WechatMessage:
    msg_id: int
    talker: str          # Conversation identifier (wxid or group@chatroom)
    sender: str          # Message sender wxid
    sender_name: str     # Resolved sender display name
    type: int            # Message type (1=text, 3=image, 34=voice, 43=video, 47=emoji, 10000=system)
    content: str         # Message text content
    timestamp: int       # Unix timestamp (seconds or ms depending on source)
    is_self: bool        # Whether this is sent by the account owner


# WeChat message types
MSG_TYPE_TEXT = 1
MSG_TYPE_IMAGE = 3
MSG_TYPE_VOICE = 34
MSG_TYPE_VIDEO = 43
MSG_TYPE_EMOJI = 47
MSG_TYPE_SYSTEM = 10000


# ── Column discovery ─────────────────────────────────────────────────────────


def _discover_columns(conn: sqlite3.Connection, table: str) -> dict[str, str]:
    """Discover columns in a table and map them to standard names.

    Returns a dict mapping standard_name -> actual_column_name for known fields.
    """
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info([{table}])").fetchall()]
    col_lower = {c.lower(): c for c in cols}

    mapping = {}
    # Contact table columns
    for std, candidates in [
        ("username", ["userName", "username", "strUsrName", "usrName"]),
        ("alias", ["alias", "strAlias"]),
        ("nickname", ["nickName", "nickname", "strNickName", "dbContactRemark"]),
        ("remark", ["remark", "strRemark", "dbContactRemark"]),
    ]:
        for c in candidates:
            if c.lower() in col_lower:
                mapping[std] = col_lower[c.lower()]
                break

    # Message table columns
    for std, candidates in [
        ("talker", ["talker", "strTalker", "userName", "username"]),
        ("content", ["content", "message_content", "strContent", "msgContent"]),
        ("type", ["type", "local_type", "nMsgType", "msgType"]),
        ("createtime", ["createTime", "create_time", "nCreateTime", "timestamp"]),
        ("localid", ["localId", "local_id", "msgSeq", "nMsgId"]),
        ("sender", ["real_sender_id", "msgSource", "strSender", "sender"]),
    ]:
        for c in candidates:
            if c.lower() in col_lower:
                mapping[std] = col_lower[c.lower()]
                break

    return mapping


# ── Contact extraction ──────────────────────────────────────────────────────


def list_contacts(decrypted_db_dir: str) -> list[WechatContact]:
    """List all contacts from decrypted contact.db.

    Uses dynamic column discovery to handle schema variations.
    Groups are identified by userName ending in '@chatroom'.
    """
    contact_db = os.path.join(decrypted_db_dir, "contact", "contact.db")
    if not os.path.exists(contact_db):
        print(f"contact.db not found at {contact_db}")
        return []

    contacts = []
    conn = sqlite3.connect(contact_db)
    conn.row_factory = sqlite3.Row

    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        # Find contact table by name or by column content
        contact_table = None
        for candidate in ["contact", "rcontact", "RC"]:
            if candidate in tables:
                contact_table = candidate
                break

        if contact_table is None:
            for table in tables:
                try:
                    col_map = _discover_columns(conn, table)
                    if "username" in col_map:
                        contact_table = table
                        break
                except sqlite3.OperationalError:
                    continue

        if contact_table is None:
            print(f"No contact table found. Available tables: {tables}")
            return []

        col_map = _discover_columns(conn, contact_table)
        rows = conn.execute(f"SELECT * FROM [{contact_table}]").fetchall()

        for row in rows:
            try:
                d = dict(row)
                wxid = d.get(col_map.get("username", ""), "")
                if not wxid:
                    continue
                # Skip service accounts and system contacts
                if wxid.startswith(("gh_", "weixin", "medianote", "floatbottle", "filehelper")):
                    continue
                is_group = wxid.endswith("@chatroom") or wxid.endswith("@chatroom_staff")
                contacts.append(WechatContact(
                    wxid=wxid,
                    alias=d.get(col_map.get("alias", ""), "") or "",
                    nickname=d.get(col_map.get("nickname", ""), "") or "",
                    remark=d.get(col_map.get("remark", ""), "") or "",
                    is_group=is_group,
                ))
            except Exception:
                continue
    finally:
        conn.close()

    return contacts


def list_sessions(decrypted_db_dir: str) -> list[dict]:
    """List recent conversation sessions from session.db.

    Returns session list sorted by last message time (most recent first).
    """
    session_db = os.path.join(decrypted_db_dir, "session", "session.db")
    if not os.path.exists(session_db):
        print(f"session.db not found at {session_db}")
        return []

    sessions = []
    conn = sqlite3.connect(session_db)
    conn.row_factory = sqlite3.Row

    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        session_table = None
        for candidate in ["SessionTable", "session", "rsession", "Session"]:
            if candidate in tables:
                session_table = candidate
                break

        if session_table is None:
            for table in tables:
                try:
                    col_map = _discover_columns(conn, table)
                    if "username" in col_map:
                        session_table = table
                        break
                except sqlite3.OperationalError:
                    continue

        if session_table is None:
            print(f"No session table found. Available tables: {tables}")
            return []

        col_map = _discover_columns(conn, session_table)
        rows = conn.execute(f"SELECT * FROM [{session_table}]").fetchall()

        for row in rows:
            try:
                d = dict(row)
                wxid = d.get(col_map.get("username", ""), "")
                if not wxid:
                    continue
                sessions.append({
                    "wxid": wxid,
                    "nickname": d.get(col_map.get("nickname", ""), "") or "",
                    "last_msg_time": d.get("nTime", 0) or d.get("createTime", 0) or 0,
                    "unread_count": d.get("unreadCount", 0) or d.get("nUnReadCount", 0) or 0,
                })
            except Exception:
                continue
    finally:
        conn.close()

    sessions.sort(key=lambda s: s["last_msg_time"], reverse=True)
    return sessions


# ── Message extraction ───────────────────────────────────────────────────────


def _msg_table_name_for_username(username: str) -> str:
    """Compute the per-user message table name (WeChat 4.x convention)."""
    table_hash = hashlib.md5(username.encode()).hexdigest()
    return f"Msg_{table_hash}"


def _discover_message_tables(conn: sqlite3.Connection) -> list[str]:
    """Discover message tables in a message database.

    WeChat 4.x Linux stores messages in tables named Msg_<hash> (one per contact).
    Also checks for a unified MSG table.
    """
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    msg_tables = []
    for table in tables:
        if table.startswith("Msg_") or table.upper() == "MSG":
            msg_tables.append(table)
            continue
        # Check for message-like columns
        try:
            col_map = _discover_columns(conn, table)
            if "content" in col_map and ("talker" in col_map or "createtime" in col_map):
                msg_tables.append(table)
        except sqlite3.OperationalError:
            continue

    return msg_tables


def _parse_sender_from_content(content: str, is_group: bool) -> tuple[str, str]:
    """Extract sender wxid from message content for group chats.

    Group messages may have the format: 'senderwxid:\\nactual_message'
    or the sender info is in a separate field (real_sender_id).

    Returns (sender_wxid, cleaned_content).
    """
    if not is_group:
        return "", content

    # Group message: "wxid_xxx:\nactual content"
    if ":\n" in content:
        parts = content.split("\n", 1)
        if len(parts) == 2:
            sender_candidate = parts[0].rstrip(":")
            if sender_candidate.startswith("wxid_") or "@" in sender_candidate:
                return sender_candidate, parts[1]

    return "", content


def extract_messages(
    decrypted_db_dir: str,
    contact_wxid: str = "",
    include_types: list[int] | None = None,
    max_messages: int = 0,
    start_time: str = "",
    end_time: str = "",
    self_wxid: str = "",
) -> list[WechatMessage]:
    """Extract chat messages from decrypted message databases.

    Queries message_0.db (and biz_message_0.db for business chats).
    Uses parameterized queries to prevent SQL injection.

    Args:
        decrypted_db_dir: Path to the decrypted db_storage directory.
        contact_wxid: If specified, only extract messages for this contact/group.
        include_types: Message type filter (default: text only [1]).
        max_messages: Max messages per contact (0 = unlimited).
        start_time: ISO format start time filter (empty = no filter).
        end_time: ISO format end time filter (empty = no filter).
        self_wxid: The owner's wxid for marking is_self.

    Returns:
        List of WechatMessage sorted by timestamp ascending.
    """
    if include_types is None:
        include_types = [MSG_TYPE_TEXT]

    msg_db_dir = os.path.join(decrypted_db_dir, "message")
    db_files = []
    if os.path.isdir(msg_db_dir):
        for name in sorted(os.listdir(msg_db_dir)):
            if (name.endswith(".db")
                    and "message" in name.lower()
                    and "fts" not in name.lower()
                    and "resource" not in name.lower()
                    and "revoke" not in name.lower()):
                db_files.append(os.path.join(msg_db_dir, name))

    all_messages = []
    for db_path in db_files:
        all_messages.extend(
            _extract_from_db(db_path, contact_wxid, include_types, max_messages,
                             start_time, end_time, self_wxid)
        )

    all_messages.sort(key=lambda m: m.timestamp)
    return all_messages


def _extract_from_db(
    db_path: str,
    contact_wxid: str,
    include_types: list[int],
    max_messages: int,
    start_time: str,
    end_time: str,
    self_wxid: str,
) -> list[WechatMessage]:
    """Extract messages from a single decrypted message database."""
    if not os.path.exists(db_path):
        return []

    messages = []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # Determine which tables to query
        if contact_wxid:
            # We know the target: try the per-user table first
            expected_table = _msg_table_name_for_username(contact_wxid)
            tables_in_db = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if expected_table in tables_in_db:
                msg_tables = [expected_table]
            else:
                # Fall back to scanning all message tables
                msg_tables = _discover_message_tables(conn)
        else:
            msg_tables = _discover_message_tables(conn)

        if not msg_tables:
            return []

        for table in msg_tables:
            try:
                col_map = _discover_columns(conn, table)
                if "content" not in col_map:
                    continue

                # Build parameterized query
                params = []
                conditions = []

                type_col = col_map.get("type", "type")
                placeholders = ",".join("?" for _ in include_types)
                conditions.append(f"{type_col} IN ({placeholders})")
                params.extend(include_types)

                if contact_wxid and "talker" in col_map:
                    conditions.append(f"{col_map['talker']} = ?")
                    params.append(contact_wxid)

                if start_time:
                    ts = int(datetime.fromisoformat(start_time).timestamp())
                    time_col = col_map.get("createtime", "createTime")
                    conditions.append(f"{time_col} >= ?")
                    params.append(ts)

                if end_time:
                    ts = int(datetime.fromisoformat(end_time).timestamp())
                    time_col = col_map.get("createtime", "createTime")
                    conditions.append(f"{time_col} <= ?")
                    params.append(ts)

                where = " AND ".join(conditions)
                time_col = col_map.get("createtime", "createTime")
                query = f"SELECT * FROM [{table}] WHERE {where} ORDER BY {time_col} ASC"
                if max_messages > 0:
                    query += f" LIMIT {max_messages}"

                rows = conn.execute(query, params).fetchall()

                for row in rows:
                    try:
                        d = dict(row)
                        talker = d.get(col_map.get("talker", ""), "")
                        content = d.get(col_map["content"], "") or ""
                        msg_type = d.get(col_map.get("type", "type"), 0)
                        timestamp = d.get(col_map.get("createtime", "createTime"), 0) or 0
                        msg_id = d.get(col_map.get("localid", "localId"), 0) or 0
                        sender = d.get(col_map.get("sender", "real_sender_id"), "")

                        is_group = talker.endswith("@chatroom")
                        # Parse sender for group messages
                        group_sender, cleaned_content = _parse_sender_from_content(
                            content, is_group
                        )
                        if group_sender:
                            sender = group_sender
                            content = cleaned_content

                        is_self = (sender == self_wxid) if self_wxid else False

                        # Skip empty text messages
                        if msg_type == MSG_TYPE_TEXT and not content.strip():
                            continue

                        messages.append(WechatMessage(
                            msg_id=msg_id,
                            talker=talker,
                            sender=sender,
                            sender_name="",
                            type=msg_type,
                            content=content,
                            timestamp=timestamp,
                            is_self=is_self,
                        ))
                    except Exception:
                        continue
            except sqlite3.OperationalError:
                continue
    finally:
        conn.close()

    return messages


def _resolve_sender_names(messages: list[WechatMessage], contacts: list[WechatContact]) -> None:
    """Resolve sender wxid to display names using the contact list."""
    name_map = {c.wxid: c.display_name for c in contacts}
    for msg in messages:
        if msg.sender and msg.sender in name_map:
            msg.sender_name = name_map[msg.sender]
        elif msg.talker and msg.talker in name_map:
            msg.sender_name = name_map[msg.talker]


# ── CLI entry point ──────────────────────────────────────────────────────────


def extract_and_save(config: WechatExtractConfig | None = None) -> dict[str, list[WechatMessage]]:
    """Extract messages and save to JSONL files.

    If no target contacts are specified, lists sessions for user selection.
    Saves per-contact JSONL to config.output_dir.

    Returns:
        Dict mapping contact_wxid -> list of WechatMessage.
    """
    if config is None:
        config = WechatExtractConfig()

    decrypted_dir = config.decrypted_dir
    output_dir = config.output_dir

    contacts = list_contacts(decrypted_dir)

    target_contacts = list(config.target_contacts)
    if not target_contacts:
        sessions = list_sessions(decrypted_dir)
        if not sessions:
            print("No sessions found. Check that databases are decrypted.")
            return {}

        print("\nAvailable conversations:")
        print(f"{'#':>4}  {'Name':<30}  {'wxid':<40}  {'Last Active'}")
        print("-" * 110)
        for i, s in enumerate(sessions[:50]):
            ts = s["last_msg_time"]
            # Try both seconds and ms interpretation
            if ts > 1e12:
                time_str = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M") if ts > 0 else "N/A"
            else:
                time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts > 0 else "N/A"
            name = s["nickname"] or s["wxid"]
            print(f"{i:4d}  {name:<30}  {s['wxid']:<40}  {time_str}")

        choice = input("\nEnter numbers or wxids (comma-separated), or 'all': ").strip()
        if choice.lower() == "all":
            target_contacts = [s["wxid"] for s in sessions]
        else:
            for part in choice.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part)
                    if 0 <= idx < len(sessions):
                        target_contacts.append(sessions[idx]["wxid"])
                else:
                    target_contacts.append(part)

    # Determine self_wxid from decrypted directory structure
    self_wxid = _detect_self_wxid(decrypted_dir)

    os.makedirs(output_dir, exist_ok=True)
    result = {}

    for wxid in target_contacts:
        print(f"\nExtracting messages for {wxid}...")
        messages = extract_messages(
            decrypted_dir,
            contact_wxid=wxid,
            include_types=config.include_types,
            max_messages=config.max_messages,
            start_time=config.start_time,
            end_time=config.end_time,
            self_wxid=self_wxid,
        )
        _resolve_sender_names(messages, contacts)

        safe_name = wxid.replace("/", "_").replace("\\", "_")
        out_path = os.path.join(output_dir, f"{safe_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(asdict(msg), ensure_ascii=False) + "\n")

        result[wxid] = messages
        print(f"  Saved {len(messages)} messages to {out_path}")

    return result


def _detect_self_wxid(decrypted_dir: str) -> str:
    """Try to detect the account owner's wxid from the directory structure.

    The decrypted dir is typically wechat_data/decrypted, and the original
    path is ~/Documents/xwechat_files/<wxid>/db_storage/.
    We look for a stored _db_dir path in the keys file, or try to infer
    from the directory name.
    """
    # Check if keys.json has the _db_dir metadata
    keys_path = os.path.join(os.path.dirname(decrypted_dir), "all_keys.json")
    if os.path.exists(keys_path):
        try:
            with open(keys_path) as f:
                keys = json.load(f)
            db_dir = keys.get("_db_dir", "")
            # db_dir is like /home/user/Documents/xwechat_files/wxid_xxx/db_storage
            parent = os.path.basename(os.path.dirname(db_dir))
            if parent.startswith("wxid_"):
                return parent
        except Exception:
            pass

    # Try from the all_keys.json in the default location
    for candidate in ["wechat_data/all_keys.json", "all_keys.json"]:
        if os.path.exists(candidate):
            try:
                with open(candidate) as f:
                    keys = json.load(f)
                db_dir = keys.get("_db_dir", "")
                parent = os.path.basename(os.path.dirname(db_dir))
                if parent.startswith("wxid_"):
                    return parent
            except Exception:
                pass

    print("[WARN] Could not auto-detect self wxid. "
          "Messages may not be correctly marked as is_self.")
    print("       You can set it manually in WechatExtractConfig.")
    return ""
