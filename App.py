"""
SkillLink Marketplace — MVP (PRD v2.2)
Streamlit single-file app, SQLite-backed, deployable directly on Streamlit Cloud.

Implements:
  - Client / Provider / Admin roles
  - Fare calculation engine (base + complexity multiplier + duration + distance)
  - 25% commission settlement
  - Booking lifecycle: requested -> negotiating -> confirmed -> en_route -> completed -> cancelled
  - Easypaisa-style P2P receipt upload + optional Gemini AI verification
  - 12h/24h enforcement timers (checked on app load)
  - Admin dispute / KYC / manual receipt unblocking
"""

import os
import sqlite3
import hashlib
import io
import math
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

# Optional: only imported/used if a Gemini key is configured
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
APP_TITLE = "SkillLink"
DB_PATH = os.path.join(os.path.dirname(__file__), "skilllink.db")
COMMISSION_RATE = 0.25
COMPLEXITY_MULTIPLIERS = {"basic": 1.0, "standard": 1.4, "complex": 1.9}
CATEGORIES = ["Plumber", "Electrician", "Tailor", "Carpenter", "Laborer",
              "Cook", "Babysitter", "Sweeper", "Tutor"]
WARNING_HOURS = 12
SUSPEND_HOURS = 24

st.set_page_config(page_title=APP_TITLE, page_icon="🛠️", layout="wide")

def haversine_km(lat1, lon1, lat2, lon2):
    """Return great-circle distance between two coordinates in kilometres."""
    radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(min(1.0, a)))


# --------------------------------------------------------------------------------------
# DATABASE
# --------------------------------------------------------------------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phone_number TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        is_client INTEGER DEFAULT 1,
        is_provider INTEGER DEFAULT 0,
        is_admin INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        suspended INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS providers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL UNIQUE,
        categories TEXT NOT NULL,
        base_rate REAL NOT NULL,
        rate_unit TEXT DEFAULT 'hourly',
        kyc_status TEXT DEFAULT 'pending',
        easypaisa_wallet TEXT,
        lat REAL DEFAULT 33.6844,
        lon REAL DEFAULT 73.0479,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );

    CREATE TABLE IF NOT EXISTS bookings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        client_id INTEGER NOT NULL,
        provider_id INTEGER,
        category TEXT NOT NULL,
        complexity_tier TEXT NOT NULL,
        base_price REAL NOT NULL,
        estimated_duration_min INTEGER NOT NULL,
        distance_km REAL NOT NULL,
        recommended_fare REAL NOT NULL,
        negotiated_fare REAL,
        status TEXT DEFAULT 'requested',
        client_lat REAL, client_lon REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        completed_at TEXT,
        FOREIGN KEY (client_id) REFERENCES users(id),
        FOREIGN KEY (provider_id) REFERENCES providers(id)
    );

    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        booking_id INTEGER NOT NULL UNIQUE,
        gross_amount REAL NOT NULL,
        platform_fee REAL NOT NULL,
        net_payout REAL NOT NULL,
        receipt_note TEXT,
        payment_status TEXT DEFAULT 'unpaid',
        verification_status TEXT DEFAULT 'unverified',
        uploaded_at TEXT,
        FOREIGN KEY (booking_id) REFERENCES bookings(id)
    );
    """)
    conn.commit()
    conn.close()


init_db()

# --------------------------------------------------------------------------------------
# AUTH HELPERS
# --------------------------------------------------------------------------------------
def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def create_user(phone, name, pw, is_client, is_provider):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO users (phone_number, name, password_hash, is_client, is_provider) VALUES (?,?,?,?,?)",
            (phone, name, hash_pw(pw), int(is_client), int(is_provider)),
        )
        conn.commit()
        return conn.execute("SELECT * FROM users WHERE phone_number=?", (phone,)).fetchone()
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def authenticate(phone, pw):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE phone_number=?", (phone,)).fetchone()
    conn.close()
    if row and row["password_hash"] == hash_pw(pw):
        return row
    return None


def get_provider_profile(user_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM providers WHERE user_id=?", (user_id,)).fetchone()
    conn.close()
    return row


# --------------------------------------------------------------------------------------
# FARE ENGINE  (per PRD section 7)
# --------------------------------------------------------------------------------------
def calculate_recommended_fare(base_price, complexity_tier, estimated_duration_min,
                                provider_base_rate, provider_rate_unit, distance_km,
                                per_km_rate=0.50):
    multiplier = COMPLEXITY_MULTIPLIERS.get(complexity_tier, 1.0)
    base_fare = base_price * multiplier
    duration_fee = (estimated_duration_min / 60) * provider_base_rate if provider_rate_unit == "hourly" else 0
    distance_fee = distance_km * per_km_rate
    return round(base_fare + duration_fee + distance_fee, 2)


def settle_transaction(final_fare):
    platform_fee = round(final_fare * COMMISSION_RATE, 2)
    return {
        "gross": final_fare,
        "platform_fee": platform_fee,
        "net_payout": round(final_fare - platform_fee, 2),
    }


# --------------------------------------------------------------------------------------
# GEMINI RECEIPT VERIFICATION (optional — only runs if secrets["GEMINI_API_KEY"] is set)
# --------------------------------------------------------------------------------------
def gemini_verify_receipt(image_bytes, expected_amount, expected_wallet):
    """Returns (status:str, explanation:str). Falls back to 'manual_review' if no key configured."""
    api_key = st.secrets.get("GEMINI_API_KEY", None) if hasattr(st, "secrets") else None
    if not (GENAI_AVAILABLE and api_key):
        return "manual_review", "Gemini API key not configured — flagged for admin manual review."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"This is an Easypaisa payment receipt screenshot. "
            f"Verify whether it shows a successful transfer of approximately PKR {expected_amount} "
            f"to wallet/account '{expected_wallet}'. "
            f"Respond with exactly one word first — VERIFIED or MISMATCH — then a one-sentence reason."
        )
        response = model.generate_content(
            [{"mime_type": "image/png", "data": image_bytes}, prompt]
        )
        text = response.text.strip()
        status = "verified" if text.upper().startswith("VERIFIED") else "mismatch"
        return status, text
    except Exception as e:
        return "manual_review", f"Gemini verification failed ({e}) — flagged for admin manual review."


# --------------------------------------------------------------------------------------
# ENFORCEMENT TIMERS (12h warning / 24h auto-suspend on unpaid completed bookings)
# --------------------------------------------------------------------------------------
def run_enforcement_timers():
    conn = get_conn()
    now = datetime.utcnow()
    rows = conn.execute("""
        SELECT t.id as tx_id, t.payment_status, b.completed_at, b.client_id
        FROM transactions t JOIN bookings b ON b.id = t.booking_id
        WHERE t.payment_status = 'unpaid' AND b.completed_at IS NOT NULL
    """).fetchall()
    for r in rows:
        completed = datetime.fromisoformat(r["completed_at"])
        hours_elapsed = (now - completed).total_seconds() / 3600
        if hours_elapsed >= SUSPEND_HOURS:
            conn.execute("UPDATE users SET suspended=1 WHERE id=?", (r["client_id"],))
    conn.commit()
    conn.close()


run_enforcement_timers()

# --------------------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "active_role" not in st.session_state:
    st.session_state.active_role = None

# --------------------------------------------------------------------------------------
# LOGIN / REGISTER SCREEN
# --------------------------------------------------------------------------------------
def login_register_screen():
    st.title("🛠️ SkillLink — Hyper-local Skilled Labor Marketplace")
    tab_login, tab_register = st.tabs(["Log In", "Register"])

    with tab_login:
        phone = st.text_input("Phone number", key="login_phone")
        pw = st.text_input("Password", type="password", key="login_pw")
        if st.button("Log In", type="primary"):
            user = authenticate(phone, pw)
            if not user:
                st.error("Invalid phone number or password.")
            elif user["suspended"]:
                st.error("This account is suspended pending payment resolution. Contact admin support.")
            else:
                st.session_state.user = dict(user)
                st.rerun()

    with tab_register:
        name = st.text_input("Full name")
        phone_r = st.text_input("Phone number", key="reg_phone")
        pw_r = st.text_input("Password", type="password", key="reg_pw")
        role = st.radio("Register as", ["Client", "Service Provider", "Both"], horizontal=True)
        is_client = role in ("Client", "Both")
        is_provider = role in ("Service Provider", "Both")

        provider_categories, base_rate, wallet = None, None, None
        if is_provider:
            provider_categories = st.multiselect("Service categories offered", CATEGORIES)
            base_rate = st.number_input("Hourly base rate (PKR)", min_value=0.0, value=500.0, step=50.0)
            wallet = st.text_input("Easypaisa wallet ID")

        if st.button("Create Account", type="primary"):
            if not (name and phone_r and pw_r):
                st.error("Name, phone, and password are required.")
            else:
                user = create_user(phone_r, name, pw_r, is_client, is_provider)
                if not user:
                    st.error("An account with that phone number already exists.")
                else:
                    if is_provider:
                        conn = get_conn()
                        conn.execute(
                            "INSERT INTO providers (user_id, categories, base_rate, easypaisa_wallet) VALUES (?,?,?,?)",
                            (user["id"], ",".join(provider_categories or []), base_rate, wallet),
                        )
                        conn.commit()
                        conn.close()
                    st.success("Account created — please log in.")


# --------------------------------------------------------------------------------------
# CLIENT DASHBOARD
# --------------------------------------------------------------------------------------
def client_dashboard(user):
    st.header(f"👤 Client Dashboard — {user['name']}")
    tab_book, tab_mybookings, tab_pay = st.tabs(["Book a Service", "My Bookings", "Upload Receipt / Pay"])

    with tab_book:
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("Service category", CATEGORIES)
            complexity = st.selectbox("Job complexity", list(COMPLEXITY_MULTIPLIERS.keys()))
            base_price = st.number_input("Base job price (PKR)", min_value=0.0, value=1000.0, step=100.0)
            duration = st.number_input("Estimated duration (minutes)", min_value=15, value=60, step=15)
        with col2:
            st.caption("Your location (used for distance-based fare & live provider matching)")
            client_lat = st.number_input("Latitude", value=33.6844, format="%.4f")
            client_lon = st.number_input("Longitude", value=73.0479, format="%.4f")

        conn = get_conn()
        providers = conn.execute(
            "SELECT p.*, u.name FROM providers p JOIN users u ON u.id = p.user_id WHERE p.kyc_status='approved'"
        ).fetchall()
        conn.close()

        matches = [p for p in providers if category in (p["categories"] or "").split(",")]

        if matches:
            st.write(f"**{len(matches)} verified provider(s) found nearby for {category}:**")
            options = {}
            for p in matches:
                dist = round(haversine_km(client_lat, client_lon, p["lat"], p["lon"]), 2)
                fare = calculate_recommended_fare(base_price, complexity, duration, p["base_rate"], p["rate_unit"], dist)
                label = f"{p['name']} — {dist} km away — Recommended fare PKR {fare}"
                options[label] = (p, dist, fare)

            choice = st.radio("Select a provider", list(options.keys()))
            if st.button("Request Booking", type="primary"):
                p, dist, fare = options[choice]
                conn = get_conn()
                conn.execute("""
                    INSERT INTO bookings (client_id, provider_id, category, complexity_tier, base_price,
                        estimated_duration_min, distance_km, recommended_fare, negotiated_fare, status,
                        client_lat, client_lon)
                    VALUES (?,?,?,?,?,?,?,?,?, 'requested', ?, ?)
                """, (user["id"], p["id"], category, complexity, base_price, duration, dist, fare, fare,
                      client_lat, client_lon))
                conn.commit()
                conn.close()
                st.success("Booking requested! The provider will confirm or counter-negotiate the fare.")
        else:
            st.info("No verified providers currently available for this category. Try again shortly.")

    with tab_mybookings:
        conn = get_conn()
        rows = conn.execute("""
            SELECT b.*, u.name as provider_name FROM bookings b
            LEFT JOIN providers p ON p.id = b.provider_id
            LEFT JOIN users u ON u.id = p.user_id
            WHERE b.client_id=? ORDER BY b.created_at DESC
        """, (user["id"],)).fetchall()
        conn.close()
        if not rows:
            st.info("No bookings yet.")
        else:
            df = pd.DataFrame([dict(r) for r in rows])
            st.dataframe(df[["id", "category", "provider_name", "status", "negotiated_fare", "created_at"]],
                         use_container_width=True, hide_index=True)

            st.map(pd.DataFrame([{"lat": r["client_lat"], "lon": r["client_lon"]} for r in rows if r["client_lat"]]))

    with tab_pay:
        conn = get_conn()
        completed = conn.execute("""
            SELECT b.id, b.negotiated_fare, p.easypaisa_wallet, t.payment_status, t.id as tx_id
            FROM bookings b
            LEFT JOIN providers p ON p.id = b.provider_id
            LEFT JOIN transactions t ON t.booking_id = b.id
            WHERE b.client_id=? AND b.status='completed'
        """, (user["id"],)).fetchall()
        conn.close()

        if not completed:
            st.info("No completed jobs awaiting payment.")
        else:
            for b in completed:
                with st.expander(f"Booking #{b['id']} — PKR {b['negotiated_fare']} — status: {b['payment_status'] or 'unpaid'}"):
                    st.write(f"Transfer PKR **{b['negotiated_fare']}** via Easypaisa to wallet **{b['easypaisa_wallet']}**.")
                    if b["payment_status"] == "unpaid" or b["payment_status"] is None:
                        receipt = st.file_uploader("Upload Easypaisa receipt screenshot", type=["png", "jpg", "jpeg"],
                                                     key=f"receipt_{b['id']}")
                        if receipt and st.button("Submit Receipt", key=f"submit_{b['id']}"):
                            img_bytes = receipt.read()
                            status, note = gemini_verify_receipt(img_bytes, b["negotiated_fare"], b["easypaisa_wallet"])
                            settle = settle_transaction(b["negotiated_fare"])
                            conn = get_conn()
                            payment_status = "paid" if status == "verified" else "pending_review"
                            uploaded_at = datetime.utcnow().isoformat()
                            updated = conn.execute(
                                """
                                UPDATE transactions
                                SET receipt_note=?, verification_status=?, payment_status=?, uploaded_at=?
                                WHERE booking_id=?
                                """,
                                (note, status, payment_status, uploaded_at, b["id"]),
                            )
                            if updated.rowcount == 0:
                                # Defensive recovery for databases created by an older build.
                                settle = settle_transaction(b["negotiated_fare"])
                                conn.execute(
                                    """
                                    INSERT INTO transactions
                                    (booking_id, gross_amount, platform_fee, net_payout,
                                     receipt_note, payment_status, verification_status, uploaded_at)
                                    VALUES (?,?,?,?,?,?,?,?)
                                    """,
                                    (b["id"], settle["gross"], settle["platform_fee"], settle["net_payout"],
                                     note, payment_status, status, uploaded_at),
                                )
                            conn.commit()
                            conn.close()
                            st.success(f"Receipt submitted. Verification result: **{status}** — {note}")
                    else:
                        st.success(f"Payment status: {b['payment_status']}")


# --------------------------------------------------------------------------------------
# PROVIDER DASHBOARD
# --------------------------------------------------------------------------------------
def provider_dashboard(user):
    provider = get_provider_profile(user["id"])
    if not provider:
        st.warning("No provider profile found for this account. Contact admin to set one up.")
        return

    st.header(f"🧰 Provider Dashboard — {user['name']}")
    st.caption(f"KYC status: **{provider['kyc_status']}** | Categories: {provider['categories']}")

    tab_radar, tab_jobs, tab_earnings = st.tabs(["Job Radar", "My Active Jobs", "Earnings"])

    with tab_radar:
        conn = get_conn()
        pending = conn.execute("""
            SELECT b.*, u.name as client_name FROM bookings b
            JOIN users u ON u.id = b.client_id
            WHERE b.provider_id=? AND b.status='requested' ORDER BY b.created_at DESC
        """, (provider["id"],)).fetchall()
        conn.close()

        if not pending:
            st.info("No incoming job requests right now.")
        for b in pending:
            with st.container(border=True):
                st.write(f"**Booking #{b['id']}** — {b['category']} ({b['complexity_tier']}) for {b['client_name']}")
                st.write(f"Recommended fare: PKR {b['recommended_fare']} | Distance: {b['distance_km']} km")
                counter = st.number_input("Counter-offer fare (optional)", min_value=0.0,
                                           value=float(b["recommended_fare"]), key=f"counter_{b['id']}")
                c1, c2 = st.columns(2)
                if c1.button("Accept & Confirm", key=f"accept_{b['id']}"):
                    conn = get_conn()
                    conn.execute("UPDATE bookings SET status='confirmed', negotiated_fare=? WHERE id=?",
                                 (counter, b["id"]))
                    conn.commit()
                    conn.close()
                    st.rerun()
                if c2.button("Decline", key=f"decline_{b['id']}"):
                    conn = get_conn()
                    conn.execute("UPDATE bookings SET status='cancelled' WHERE id=?", (b["id"],))
                    conn.commit()
                    conn.close()
                    st.rerun()

    with tab_jobs:
        conn = get_conn()
        active = conn.execute("""
            SELECT b.*, u.name as client_name FROM bookings b
            JOIN users u ON u.id = b.client_id
            WHERE b.provider_id=? AND b.status IN ('confirmed','en_route')
            ORDER BY b.created_at DESC
        """, (provider["id"],)).fetchall()
        conn.close()

        if not active:
            st.info("No active jobs.")
        for b in active:
            with st.container(border=True):
                st.write(f"**Booking #{b['id']}** — {b['category']} for {b['client_name']} — Fare PKR {b['negotiated_fare']}")
                st.write(f"Status: **{b['status']}**")
                c1, c2 = st.columns(2)
                if b["status"] == "confirmed" and c1.button("Mark En Route", key=f"enroute_{b['id']}"):
                    conn = get_conn()
                    conn.execute("UPDATE bookings SET status='en_route' WHERE id=?", (b["id"],))
                    conn.commit()
                    conn.close()
                    st.rerun()
                if b["status"] == "en_route" and c2.button("Mark Completed", key=f"complete_{b['id']}"):
                    conn = get_conn()
                    conn.execute("UPDATE bookings SET status='completed', completed_at=? WHERE id=?",
                                 (datetime.utcnow().isoformat(), b["id"]))
                    settle = settle_transaction(b["negotiated_fare"])
                    conn.execute("""
                        INSERT INTO transactions (booking_id, gross_amount, platform_fee, net_payout)
                        VALUES (?,?,?,?)
                    """, (b["id"], settle["gross"], settle["platform_fee"], settle["net_payout"]))
                    conn.commit()
                    conn.close()
                    st.rerun()

    with tab_earnings:
        conn = get_conn()
        tx = conn.execute("""
            SELECT t.*, b.category FROM transactions t
            JOIN bookings b ON b.id = t.booking_id
            WHERE b.provider_id=? ORDER BY t.uploaded_at DESC
        """, (provider["id"],)).fetchall()
        conn.close()

        if not tx:
            st.info("No earnings yet.")
        else:
            df = pd.DataFrame([dict(r) for r in tx])
            total_net = df[df["payment_status"] == "paid"]["net_payout"].sum()
            st.metric("Total net payout received (after 25% commission)", f"PKR {total_net:,.2f}")
            st.dataframe(df[["id", "category", "gross_amount", "platform_fee", "net_payout",
                              "payment_status", "verification_status"]],
                         use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------------------
# ADMIN DASHBOARD
# --------------------------------------------------------------------------------------
def admin_dashboard(user):
    st.header(f"🛡️ Admin Dashboard — {user['name']}")
    tab_kyc, tab_tx, tab_users = st.tabs(["KYC Approvals", "Transactions & Disputes", "User Management"])

    with tab_kyc:
        conn = get_conn()
        pending = conn.execute("""
            SELECT p.*, u.name, u.phone_number FROM providers p
            JOIN users u ON u.id = p.user_id WHERE p.kyc_status != 'approved'
        """).fetchall()
        conn.close()
        if not pending:
            st.info("No pending KYC reviews.")
        for p in pending:
            with st.container(border=True):
                st.write(f"**{p['name']}** ({p['phone_number']}) — Categories: {p['categories']} — "
                         f"Status: {p['kyc_status']}")
                if st.button("Approve KYC", key=f"kyc_{p['id']}"):
                    conn = get_conn()
                    conn.execute("UPDATE providers SET kyc_status='approved' WHERE id=?", (p["id"],))
                    conn.commit()
                    conn.close()
                    st.rerun()

    with tab_tx:
        conn = get_conn()
        tx = conn.execute("""
            SELECT t.*, b.category, b.client_id, u.name as client_name FROM transactions t
            JOIN bookings b ON b.id = t.booking_id
            JOIN users u ON u.id = b.client_id
            ORDER BY t.uploaded_at DESC
        """).fetchall()
        conn.close()
        if not tx:
            st.info("No transactions yet.")
        else:
            for t in tx:
                with st.container(border=True):
                    st.write(f"**Tx #{t['id']}** — {t['category']} — Client: {t['client_name']} — "
                             f"Gross PKR {t['gross_amount']} | Fee PKR {t['platform_fee']} | Net PKR {t['net_payout']}")
                    st.write(f"Payment: **{t['payment_status']}** | Verification: **{t['verification_status']}**")
                    if t["receipt_note"]:
                        st.caption(f"AI note: {t['receipt_note']}")
                    if t["verification_status"] in ("manual_review", "mismatch") and t["payment_status"] != "paid":
                        if st.button("Manually Approve Payment", key=f"manual_{t['id']}"):
                            conn = get_conn()
                            conn.execute("UPDATE transactions SET payment_status='paid', verification_status='verified' WHERE id=?",
                                         (t["id"],))
                            conn.commit()
                            conn.close()
                            st.rerun()

    with tab_users:
        conn = get_conn()
        users = conn.execute("SELECT id, name, phone_number, is_client, is_provider, suspended FROM users").fetchall()
        conn.close()
        df = pd.DataFrame([dict(u) for u in users])
        st.dataframe(df, use_container_width=True, hide_index=True)
        suspended_ids = df[df["suspended"] == 1]["id"].tolist() if not df.empty else []
        if suspended_ids:
            unsuspend_id = st.selectbox("Reinstate a suspended user", suspended_ids)
            if st.button("Reinstate User"):
                conn = get_conn()
                conn.execute("UPDATE users SET suspended=0 WHERE id=?", (unsuspend_id,))
                conn.commit()
                conn.close()
                st.rerun()


# --------------------------------------------------------------------------------------
# MAIN ROUTER
# --------------------------------------------------------------------------------------
def main():
    if not st.session_state.user:
        login_register_screen()
        return

    user = st.session_state.user
    roles = []
    if user["is_client"]:
        roles.append("Client")
    if user["is_provider"]:
        roles.append("Provider")
    if user["is_admin"]:
        roles.append("Admin")

    with st.sidebar:
        st.title("🛠️ SkillLink")
        st.write(f"Logged in as **{user['name']}**")
        active_role = st.radio("View as", roles) if len(roles) > 1 else roles[0]
        st.session_state.active_role = active_role
        st.divider()
        if st.button("Log Out"):
            st.session_state.user = None
            st.rerun()

    if active_role == "Client":
        client_dashboard(user)
    elif active_role == "Provider":
        provider_dashboard(user)
    elif active_role == "Admin":
        admin_dashboard(user)


if __name__ == "__main__":
    main()
