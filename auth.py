"""
Authentication Module — RevIQ Revenue Intelligence System
Features:
  - Sign In   : login with auto-generated username + password
  - Register  : enter name + email + role → system generates credentials
  - Activity Log : every login/logout recorded with timestamp + IP-like info
  - Roles     : admin / analyst / viewer
"""

import hashlib, hmac, random, string, streamlit as st
from datetime import datetime

# ── Helpers ───────────────────────────────────────────────────────────────────
def _hash(p: str) -> str:
    return hashlib.sha256(p.encode()).hexdigest()

def _get_db() -> dict:
    if "_reviq_users" not in st.session_state:
        st.session_state._reviq_users = {
            "admin": {
                "password": _hash("Admin@2024"),
                "role": "admin", "name": "Administrator",
                "email": "admin@reviq.com",
                "joined": datetime.now().strftime("%d %b %Y"),
            }
        }
    return st.session_state._reviq_users

def _get_log() -> list:
    if "_reviq_log" not in st.session_state:
        st.session_state._reviq_log = []
    return st.session_state._reviq_log

def _add_log(username: str, name: str, action: str, role: str):
    log = _get_log()
    log.append({
        "timestamp": datetime.now().strftime("%d %b %Y  %H:%M:%S"),
        "username":  username,
        "name":      name,
        "action":    action,
        "role":      role,
    })
    st.session_state._reviq_log = log

def _gen_username(full_name: str) -> str:
    db    = _get_db()
    parts = full_name.strip().lower().split()
    base  = "".join(p for p in parts if p.isalpha())[:14]
    for _ in range(60):
        c = base + str(random.randint(100, 999))
        if c not in db:
            return c
    return base + str(random.randint(1000, 9999))

def _gen_password(n: int = 12) -> str:
    pool = string.ascii_letters + string.digits + "@#$%&!"
    pwd  = [
        random.choice(string.ascii_uppercase),
        random.choice(string.ascii_lowercase),
        random.choice(string.digits),
        random.choice("@#$%&!"),
    ] + random.choices(pool, k=n - 4)
    random.shuffle(pwd)
    return "".join(pwd)

# ── Role permissions ──────────────────────────────────────────────────────────
ROLE_PERMISSIONS = {
    "admin":   {"steps": [0,1,2,3,4,5,6,7,8], "can_train": True,  "can_download": True,  "can_reset": True,  "label": "Admin",   "color": "#ef4444", "bg": "rgba(239,68,68,0.1)"},
    "analyst": {"steps": [0,1,2,3,4,5,6,7,8], "can_train": True,  "can_download": True,  "can_reset": True,  "label": "Analyst", "color": "#6366f1", "bg": "rgba(99,102,241,0.1)"},
    "viewer":  {"steps": [1,3,4,6],            "can_train": False, "can_download": False, "can_reset": False, "label": "Viewer",  "color": "#10b981", "bg": "rgba(16,185,129,0.1)"},
}

# ── Auth state helpers ────────────────────────────────────────────────────────
def get_current_user() -> dict | None:
    return st.session_state.get("auth_user")

def is_logged_in() -> bool:
    return st.session_state.get("auth_logged_in", False)

def get_role() -> str:
    u = get_current_user()
    return u["role"] if u else "viewer"

def get_permissions() -> dict:
    return ROLE_PERMISSIONS.get(get_role(), ROLE_PERMISSIONS["viewer"])

def can_access_step(i: int) -> bool:
    return i in get_permissions()["steps"]

def logout():
    u = get_current_user()
    if u:
        _add_log(u["username"], u["name"], "Logout", u["role"])
    st.session_state.auth_logged_in = False
    st.session_state.auth_user      = None
    st.session_state.auth_attempts  = 0

# ── Shared CSS ────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.stApp { background: #080b12 !important; }
.main .block-container { max-width: 480px !important; margin: 0 auto !important; padding-top: 5vh !important; }
div[data-testid="stForm"] {
    background: #111827; border: 1px solid #1f2937;
    border-radius: 16px; padding: 1.8rem 2rem 1.5rem;
}
div[data-testid="stTextInput"] input {
    background: #1f2937 !important; border: 1px solid #374151 !important;
    border-radius: 8px !important; color: #f9fafb !important;
    font-family: 'Space Grotesk',sans-serif !important;
    font-size: 14px !important; padding: 10px 14px !important;
}
div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label { color: #9ca3af !important; font-size: 12px !important; }
div[data-testid="stSelectbox"] > div > div {
    background: #1f2937 !important; border: 1px solid #374151 !important;
    border-radius: 8px !important; color: #f9fafb !important;
}
div[data-testid="stForm"] .stButton > button {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color: #fff !important; border: none !important; border-radius: 8px !important;
    font-size: 14px !important; font-weight: 600 !important;
    padding: 12px !important; width: 100% !important; margin-top: 8px !important;
}
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
"""

def _logo():
    st.markdown("""
<div style='text-align:center;margin-bottom:1.6rem;'>
  <div style='width:52px;height:52px;background:linear-gradient(135deg,#6366f1,#06b6d4);
              border-radius:13px;display:flex;align-items:center;justify-content:center;
              font-size:24px;margin:0 auto 12px;'>📊</div>
  <div style='font-size:22px;font-weight:700;color:#f9fafb;letter-spacing:-0.03em;
              font-family:"Space Grotesk",sans-serif;'>RevIQ</div>
  <div style='font-size:11px;color:#4b5563;margin-top:3px;
              font-family:"JetBrains Mono",monospace;letter-spacing:0.05em;'>
    Revenue Intelligence System
  </div>
</div>
""", unsafe_allow_html=True)

def _err(msg):
    st.markdown(f"""
<div style='background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);
            border-left:3px solid #ef4444;border-radius:8px;padding:11px 14px;
            color:#ef4444;font-size:13px;margin-top:10px;font-family:"Space Grotesk",sans-serif;'>
  {msg}
</div>""", unsafe_allow_html=True)

def _ok(msg):
    st.markdown(f"""
<div style='background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);
            border-left:3px solid #10b981;border-radius:8px;padding:11px 14px;
            color:#10b981;font-size:13px;margin-top:10px;font-family:"Space Grotesk",sans-serif;'>
  {msg}
</div>""", unsafe_allow_html=True)

# ── Main auth page ────────────────────────────────────────────────────────────
def show_login_page():
    st.markdown(_CSS, unsafe_allow_html=True)

    # Init state
    for k, v in {"auth_logged_in": False, "auth_user": None,
                 "auth_attempts": 0, "auth_error": "",
                 "auth_tab": "signin", "reg_creds": None}.items():
        if k not in st.session_state:
            st.session_state[k] = v

    _logo()

    # ── Tab buttons ───────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        active_si = st.session_state.auth_tab == "signin"
        si_style  = ("background:rgba(99,102,241,0.15);color:#818cf8;"
                     "border:1px solid rgba(99,102,241,0.35);") if active_si else \
                    "background:transparent;color:#6b7280;border:1px solid #1f2937;"
        if st.button("🔑  Sign In", key="tab_si", use_container_width=True):
            st.session_state.auth_tab    = "signin"
            st.session_state.auth_error  = ""
            st.session_state.reg_creds   = None
            st.rerun()
        st.markdown(f"<style>div[data-testid='column']:first-child .stButton>button"
                    f"{{{si_style}border-radius:8px;font-weight:600;font-size:13px;}}</style>",
                    unsafe_allow_html=True)
    with c2:
        active_rg = st.session_state.auth_tab == "register"
        rg_style  = ("background:rgba(99,102,241,0.15);color:#818cf8;"
                     "border:1px solid rgba(99,102,241,0.35);") if active_rg else \
                    "background:transparent;color:#6b7280;border:1px solid #1f2937;"
        if st.button("📝  Register", key="tab_rg", use_container_width=True):
            st.session_state.auth_tab    = "register"
            st.session_state.auth_error  = ""
            st.session_state.reg_creds   = None
            st.rerun()
        st.markdown(f"<style>div[data-testid='column']:last-child .stButton>button"
                    f"{{{rg_style}border-radius:8px;font-weight:600;font-size:13px;}}</style>",
                    unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # SIGN IN
    # ════════════════════════════════════════════════════════
    if st.session_state.auth_tab == "signin":

        if st.session_state.auth_attempts >= 5:
            st.markdown("""
<div style='background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);
            border-radius:10px;padding:14px;text-align:center;color:#ef4444;font-size:13px;'>
  🔒 Too many failed attempts. Please restart the app.
</div>""", unsafe_allow_html=True)
            return

        with st.form("form_signin"):
            st.markdown("<div style='font-size:15px;font-weight:600;color:#f9fafb;"
                        "margin-bottom:1rem;'>Welcome back</div>", unsafe_allow_html=True)
            uname = st.text_input("Username", placeholder="Your auto-generated username")
            pwd   = st.text_input("Password", type="password",
                                  placeholder="Your auto-generated password")
            sub   = st.form_submit_button("Sign In →")

            if sub:
                if not uname or not pwd:
                    st.session_state.auth_error = "Please fill in both fields."
                else:
                    db = _get_db()
                    user = db.get(uname)
                    if user and hmac.compare_digest(user["password"], _hash(pwd)):
                        st.session_state.auth_logged_in = True
                        st.session_state.auth_user = {
                            "username": uname,
                            "name":     user["name"],
                            "email":    user["email"],
                            "role":     user["role"],
                            "login_at": datetime.now().strftime("%d %b %Y, %H:%M"),
                        }
                        st.session_state.auth_attempts = 0
                        st.session_state.auth_error    = ""
                        _add_log(uname, user["name"], "Login", user["role"])
                        st.rerun()
                    else:
                        st.session_state.auth_attempts += 1
                        left = 5 - st.session_state.auth_attempts
                        st.session_state.auth_error = (
                            f"Invalid credentials. {left} attempt{'s' if left != 1 else ''} left."
                        )

        if st.session_state.auth_error:
            _err(st.session_state.auth_error)

        st.markdown("""
<div style='margin-top:1.2rem;padding:11px 14px;background:#0d1117;border:1px solid #1f2937;
            border-radius:10px;font-size:12px;color:#4b5563;text-align:center;
            font-family:"JetBrains Mono",monospace;'>
  New here? &nbsp;<span style='color:#818cf8;'>Click "Register" above to create an account</span>
</div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # REGISTER
    # ════════════════════════════════════════════════════════
    else:
        # Show generated credentials after successful registration
        if st.session_state.reg_creds:
            c = st.session_state.reg_creds
            role_color = ROLE_PERMISSIONS[c["role"]]["color"]
            st.markdown(f"""
<div style='background:#071e12;border:1px solid #134d28;border-radius:14px;padding:22px;'>
  <div style='font-size:14px;font-weight:700;color:#10b981;margin-bottom:4px;'>
    ✅ Account created!
  </div>
  <div style='font-size:12px;color:#6b7280;margin-bottom:16px;'>
    Save these credentials now — password is shown only once.
  </div>

  <div style='background:#040d08;border-radius:10px;padding:16px 18px;
              font-family:"JetBrains Mono",monospace;'>
    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
      <span style='font-size:10px;color:#374151;text-transform:uppercase;letter-spacing:0.1em;'>Full Name</span>
      <span style='font-size:13px;font-weight:600;color:#d1fae5;'>{c["name"]}</span>
    </div>
    <div style='height:0.5px;background:#0f2d1a;margin-bottom:10px;'></div>
    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
      <span style='font-size:10px;color:#374151;text-transform:uppercase;letter-spacing:0.1em;'>Username</span>
      <span style='font-size:15px;font-weight:700;color:#f9fafb;letter-spacing:0.06em;'>{c["username"]}</span>
    </div>
    <div style='height:0.5px;background:#0f2d1a;margin-bottom:10px;'></div>
    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
      <span style='font-size:10px;color:#374151;text-transform:uppercase;letter-spacing:0.1em;'>Password</span>
      <span style='font-size:15px;font-weight:700;color:#f9fafb;letter-spacing:0.1em;'>{c["password"]}</span>
    </div>
    <div style='height:0.5px;background:#0f2d1a;margin-bottom:10px;'></div>
    <div style='display:flex;justify-content:space-between;align-items:center;'>
      <span style='font-size:10px;color:#374151;text-transform:uppercase;letter-spacing:0.1em;'>Role</span>
      <span style='font-size:12px;font-weight:700;color:{role_color};
                   text-transform:uppercase;letter-spacing:0.07em;'>{c["role"]}</span>
    </div>
  </div>

  <div style='margin-top:12px;font-size:11px;color:#374151;
              font-family:"JetBrains Mono",monospace;text-align:center;'>
    📅 Registered {c["joined"]}
  </div>
</div>
""", unsafe_allow_html=True)

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            if st.button("→  Go to Sign In", key="go_signin", use_container_width=True):
                st.session_state.auth_tab   = "signin"
                st.session_state.reg_creds  = None
                st.session_state.auth_error = ""
                st.rerun()
            return

        # Registration form
        with st.form("form_register"):
            st.markdown("<div style='font-size:15px;font-weight:600;color:#f9fafb;"
                        "margin-bottom:1rem;'>Create your account</div>",
                        unsafe_allow_html=True)

            full_name = st.text_input("Full name *", placeholder="e.g. Spoorthy Raj")
            email     = st.text_input("Email address *", placeholder="e.g. spoorthy@email.com")
            role      = st.selectbox("Role", ["analyst", "viewer"],
                                     help="Admin accounts are managed separately.")

            st.markdown("""
<div style='padding:10px 12px;background:#0d1117;border:1px solid #1f2937;
            border-radius:8px;margin:6px 0 2px;font-size:11.5px;color:#6b7280;
            font-family:"JetBrains Mono",monospace;'>
  🎲 Username &amp; password will be auto-generated after registration
</div>""", unsafe_allow_html=True)

            sub = st.form_submit_button("Create Account →")

            if sub:
                db  = _get_db()
                err = ""
                if not full_name.strip() or len(full_name.strip()) < 2:
                    err = "Please enter your full name (at least 2 characters)."
                elif not email.strip() or "@" not in email or "." not in email:
                    err = "Please enter a valid email address."
                elif email.strip().lower() in [u["email"].lower() for u in db.values()]:
                    err = "An account with this email already exists. Please sign in."
                else:
                    uname = _gen_username(full_name)
                    pwd   = _gen_password()
                    today = datetime.now().strftime("%d %b %Y")
                    _save = {
                        "password": _hash(pwd),
                        "role":     role,
                        "name":     full_name.strip(),
                        "email":    email.strip().lower(),
                        "joined":   today,
                    }
                    db[uname] = _save
                    st.session_state._reviq_users = db
                    _add_log(uname, full_name.strip(), "Registered", role)
                    st.session_state.reg_creds  = {
                        "username": uname,
                        "password": pwd,
                        "name":     full_name.strip(),
                        "role":     role,
                        "joined":   today,
                    }
                    st.session_state.auth_error = ""
                    st.rerun()

                if err:
                    st.session_state.auth_error = err

        if st.session_state.auth_error and st.session_state.auth_tab == "register":
            _err(st.session_state.auth_error)

        st.markdown("""
<div style='margin-top:1.2rem;padding:11px 14px;background:#0d1117;border:1px solid #1f2937;
            border-radius:10px;font-size:12px;color:#4b5563;text-align:center;
            font-family:"JetBrains Mono",monospace;'>
  Already have an account? &nbsp;<span style='color:#818cf8;'>Click "Sign In" above</span>
</div>""", unsafe_allow_html=True)


# ── Sidebar user card ─────────────────────────────────────────────────────────
def show_user_card():
    user  = get_current_user()
    if not user:
        return
    perms = get_permissions()
    color = perms["color"]
    bg    = perms["bg"]
    st.markdown(f"""
<div style='padding:12px 16px;background:{bg};
            border-top:1px solid {color}33;border-bottom:1px solid {color}33;'>
  <div style='display:flex;align-items:center;gap:10px;'>
    <div style='width:32px;height:32px;border-radius:50%;
                background:linear-gradient(135deg,#6366f1,#06b6d4);
                display:flex;align-items:center;justify-content:center;
                font-size:13px;font-weight:700;color:#fff;flex-shrink:0;'>
      {user["name"][0].upper()}
    </div>
    <div style='flex:1;min-width:0;'>
      <div style='font-size:13px;font-weight:600;color:#f9fafb;
                  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>
        {user["name"]}
      </div>
      <div style='font-size:10px;color:{color};font-weight:700;
                  font-family:"JetBrains Mono",monospace;text-transform:uppercase;
                  letter-spacing:0.07em;'>{perms["label"]}</div>
    </div>
  </div>
  <div style='font-size:10px;color:#374151;margin-top:6px;
              font-family:"JetBrains Mono",monospace;'>
    @{user["username"]} &nbsp;·&nbsp; {user["login_at"]}
  </div>
</div>
""", unsafe_allow_html=True)


# ── Access denied ─────────────────────────────────────────────────────────────
def show_access_denied(step_name: str = "this section"):
    st.markdown(f"""
<div style='text-align:center;padding:60px 20px;background:#111827;
            border:1px solid #1f2937;border-radius:16px;margin:20px 0;'>
  <div style='font-size:36px;margin-bottom:14px;'>🔒</div>
  <div style='font-size:16px;font-weight:600;color:#f9fafb;margin-bottom:8px;'>
    Access Restricted
  </div>
  <div style='font-size:13px;color:#4b5563;margin-bottom:16px;'>
    Your role does not have permission to access <b style="color:#9ca3af;">{step_name}</b>.
  </div>
  <div style='display:inline-block;background:rgba(239,68,68,0.1);
              border:1px solid rgba(239,68,68,0.25);border-radius:8px;
              padding:8px 16px;font-size:12px;color:#ef4444;
              font-family:"JetBrains Mono",monospace;'>
    Current role: {get_role().upper()}
  </div>
</div>
""", unsafe_allow_html=True)


# ── Activity log renderer (used in Step 9) ────────────────────────────────────
def show_activity_log():
    """Renders the full activity log table. Call this inside Step 9."""
    log = _get_log()

    st.markdown("""
<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;'>
  <div>
    <div style='font-size:18px;font-weight:700;color:#f9fafb;'>Activity Log</div>
    <div style='font-size:12px;color:#4b5563;margin-top:2px;
                font-family:"JetBrains Mono",monospace;'>
      All login / logout / registration events
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    if not log:
        st.markdown("""
<div style='text-align:center;padding:48px 20px;background:#111827;
            border:1px solid #1f2937;border-radius:14px;'>
  <div style='font-size:32px;margin-bottom:12px;'>📋</div>
  <div style='font-size:14px;color:#4b5563;'>No activity recorded yet.</div>
</div>""", unsafe_allow_html=True)
        return

    # Summary cards
    total     = len(log)
    logins    = sum(1 for e in log if e["action"] == "Login")
    logouts   = sum(1 for e in log if e["action"] == "Logout")
    registers = sum(1 for e in log if e["action"] == "Registered")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Events",   total)
    c2.metric("Logins",         logins)
    c3.metric("Logouts",        logouts)
    c4.metric("Registrations",  registers)

    st.markdown("<div style='height:1px;background:#1f2937;margin:20px 0;'></div>",
                unsafe_allow_html=True)

    # Filter
    col_f, col_s = st.columns([2, 1])
    with col_f:
        action_filter = st.selectbox("Filter by action",
                                     ["All", "Login", "Logout", "Registered"],
                                     key="log_filter")
    with col_s:
        role_filter = st.selectbox("Filter by role",
                                   ["All", "admin", "analyst", "viewer"],
                                   key="log_role_filter")

    filtered = [
        e for e in reversed(log)
        if (action_filter == "All" or e["action"] == action_filter)
        and (role_filter   == "All" or e["role"]   == role_filter)
    ]

    if not filtered:
        st.info("No events match the current filter.")
        return

    # Table header
    st.markdown("""
<div style='display:grid;grid-template-columns:180px 140px 160px 100px 90px;
            gap:8px;padding:10px 16px;background:#0d1117;border-radius:8px 8px 0 0;
            border:1px solid #1f2937;border-bottom:none;
            font-size:10px;font-weight:700;color:#374151;
            text-transform:uppercase;letter-spacing:0.1em;
            font-family:"JetBrains Mono",monospace;'>
  <div>Timestamp</div>
  <div>Username</div>
  <div>Full Name</div>
  <div>Action</div>
  <div>Role</div>
</div>""", unsafe_allow_html=True)

    action_styles = {
        "Login":      ("rgba(16,185,129,0.12)",  "#10b981"),
        "Logout":     ("rgba(239,68,68,0.10)",   "#ef4444"),
        "Registered": ("rgba(99,102,241,0.12)",  "#818cf8"),
    }

    role_colors = {
        "admin":   "#ef4444",
        "analyst": "#6366f1",
        "viewer":  "#10b981",
    }

    rows_html = ""
    for i, e in enumerate(filtered):
        bg_row    = "#111827" if i % 2 == 0 else "#0d1117"
        a_bg, a_c = action_styles.get(e["action"], ("rgba(156,163,175,0.1)", "#9ca3af"))
        r_c       = role_colors.get(e["role"], "#9ca3af")
        border_b  = "border-bottom:1px solid #1f2937;" if i < len(filtered)-1 else ""
        rows_html += f"""
<div style='display:grid;grid-template-columns:180px 140px 160px 100px 90px;
            gap:8px;padding:11px 16px;background:{bg_row};
            border-left:1px solid #1f2937;border-right:1px solid #1f2937;{border_b}
            font-size:12.5px;align-items:center;'>
  <div style='color:#6b7280;font-family:"JetBrains Mono",monospace;font-size:11px;'>
    {e["timestamp"]}
  </div>
  <div style='color:#f9fafb;font-family:"JetBrains Mono",monospace;font-weight:600;
              font-size:12px;'>
    @{e["username"]}
  </div>
  <div style='color:#9ca3af;'>{e["name"]}</div>
  <div>
    <span style='background:{a_bg};color:{a_c};font-size:10px;font-weight:700;
                 padding:3px 9px;border-radius:999px;font-family:"JetBrains Mono",monospace;
                 text-transform:uppercase;letter-spacing:0.06em;'>
      {e["action"]}
    </span>
  </div>
  <div style='color:{r_c};font-size:10px;font-weight:700;
              font-family:"JetBrains Mono",monospace;text-transform:uppercase;
              letter-spacing:0.06em;'>
    {e["role"]}
  </div>
</div>"""

    st.markdown(rows_html, unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
<div style='padding:10px 16px;background:#0d1117;border:1px solid #1f2937;
            border-top:none;border-radius:0 0 8px 8px;
            font-size:11px;color:#374151;font-family:"JetBrains Mono",monospace;
            display:flex;justify-content:space-between;'>
  <span>Showing {len(filtered)} of {total} events</span>
  <span>Most recent first</span>
</div>""", unsafe_allow_html=True)