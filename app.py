"""
Sales AI Assistant - Streamlit Application
Main entry point with 3-page navigation + duplicate detection + deletion.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta

# Set page config first
st.set_page_config(
    page_title="Sales AI Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import project modules
from database import get_db, DuplicateClientError
from models import ClientCreate, InteractionCreate
from ai_crm import extract_crm_data
from ai_followup import generate_followups
from memory import get_memory_manager
from config import check_gpu_availability

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = get_db()
if 'memory' not in st.session_state:
    st.session_state.memory = get_memory_manager()
if 'followups' not in st.session_state:
    st.session_state.followups = None
if 'crm_data' not in st.session_state:
    st.session_state.crm_data = None
if 'conversation' not in st.session_state:
    st.session_state.conversation = ""
if 'client_id' not in st.session_state:
    st.session_state.client_id = None
if 'new_client_created' not in st.session_state:
    st.session_state.new_client_created = False
if 'page' not in st.session_state:
    st.session_state.page = "Add Interaction"

# Custom CSS - Dark theme with high contrast
st.markdown("""
<style>
    /* Main headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    /* Stage badges */
    .stage-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 600;
    }
    .hot { background-color: #dc3545; color: white; }
    .warm { background-color: #fd7e14; color: black; }
    .cold { background-color: #17a2b8; color: white; }
    .neutral { background-color: #6c757d; color: white; }
    
    /* Timeline */
    .timeline-item {
        border-left: 3px solid #0d6efd;
        padding-left: 1rem;
        margin-bottom: 1rem;
    }
    
    /* GPU/CPU badges */
    .gpu-badge {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .cpu-badge {
        background-color: #6c757d;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: bold;
    }
    
    /* Make sure all text is visible */
    .stMarkdown p, .stMarkdown span, .stMarkdown div {
        color: #ffffff;
    }
    
    /* Fix expander text */
    .streamlit-expanderHeader {
        color: #ffffff !important;
    }
    
    /* Button styling */
    .stButton button {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

def render_sidebar():
    """Render navigation sidebar."""
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
        st.title("Sales AI")

        # GPU Status
        gpu_info = check_gpu_availability()
        if gpu_info['available']:
            st.markdown(f"<span style='background-color: #28a745; color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; font-weight: bold;'>🚀 GPU: {gpu_info['type'].upper()}</span>", unsafe_allow_html=True)
            if gpu_info['device_names']:
                st.caption(f"Device: {gpu_info['device_names'][0][:30]}...")
        else:
            st.markdown("<span style='background-color: #6c757d; color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; font-weight: bold;'>💻 CPU Mode</span>", unsafe_allow_html=True)

        st.markdown("---")

        # Use session state for page navigation
        page_options = ["📝 Add Interaction", "👥 Clients", "📧 Follow-ups", "📊 Dashboard"]
        default_index = page_options.index(st.session_state.page) if st.session_state.page in page_options else 0

        page = st.radio(
            "Navigation",
            page_options,
            index=default_index
        )

        # Update session state if changed
        if page != st.session_state.page:
            st.session_state.page = page
            st.rerun()

        st.markdown("---")
        st.markdown("### System Status")

        # Check Ollama status
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code == 200:
                st.success("🟢 LLM Online")
            else:
                st.error("🔴 LLM Offline")
        except:
            st.error("🔴 LLM Offline")
            st.info("Start Ollama: `ollama serve`")

        return page

def show_duplicate_warning(duplicates: list, on_continue, on_cancel):
    """Display duplicate detection warning."""
    st.error("⚠️ Potential Duplicate Clients Found")
    st.write("Similar clients already exist in the database. Please review:")

    for dup in duplicates[:3]:  # Show top 3
        cols = st.columns([3, 1, 1, 1])
        with cols[0]:
            st.write(f"**{dup['name']}** ({dup['company'] or 'No company'})")
        with cols[1]:
            st.caption(f"Name match: {dup['name_similarity']:.0f}%")
        with cols[2]:
            if dup['email_match']:
                st.caption("📧 Email exact match!")
        with cols[3]:
            st.caption(f"Score: {dup['total_score']:.0f}%")

    st.warning("These may be the same person/company. What would you like to do?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Create Anyway", type="primary", use_container_width=True):
            on_continue()
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            on_cancel()

def page_add_interaction():
    """Page 1: Add new interaction with AI processing."""
    st.markdown('<div class="main-header">📝 Add New Interaction</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Client Selection")

        # Check if we came from follow-ups page with pre-selected client
        preselected_client_id = st.session_state.get('preselected_client_id')

        # Client selection mode
        client_mode = st.radio(
            "Select mode",
            ["Existing Client", "New Client"],
            horizontal=True,
            key="client_mode",
            index=0 if not preselected_client_id else 0
        )

        client_id = None
        client_name = ""
        show_duplicates = False
        duplicate_data = None
        client_created = False

        if client_mode == "Existing Client":
            clients = st.session_state.db.get_all_clients()
            if not clients:
                st.warning("No clients found. Create one first.")
                client_mode = "New Client"
            else:
                # Create options and set default if preselected
                client_options = {f"{c.name} ({c.company or 'No company'})": c.id for c in clients}

                default_index = 0
                if preselected_client_id:
                    for idx, (name, cid) in enumerate(client_options.items()):
                        if cid == preselected_client_id:
                            default_index = idx
                            break

                selected = st.selectbox("Select client", list(client_options.keys()), index=default_index)
                client_id = client_options[selected]
                client_name = selected.split(" (")[0]
                st.session_state.client_id = client_id

                # Clear preselection after use
                if 'preselected_client_id' in st.session_state:
                    del st.session_state['preselected_client_id']

                # Show quick context
                with st.expander("View History"):
                    try:
                        history = st.session_state.memory.get_client_history(client_id)
                        st.write(f"Total interactions: {history.total_interactions}")
                        if history.last_contact:
                            st.write(f"Last contact: {history.last_contact.strftime('%Y-%m-%d')}")
                    except:
                        st.write("No history available")

        # NEW CLIENT FLOW
        if client_mode == "New Client":
            # Check if we're showing duplicate warning
            if 'duplicate_data' in st.session_state:
                show_duplicates = True
                duplicate_data = st.session_state.duplicate_data

            # Check if client was just created
            if 'new_client_id' in st.session_state and st.session_state.new_client_id:
                client_id = st.session_state.new_client_id
                client = st.session_state.db.get_client(client_id)
                if client:
                    client_name = client.name
                    client_created = True
                    st.success(f"✅ Client Created: {client.name}")
                    st.info("You can now paste conversation and click 'Process with AI'")

            # Show duplicate warning if needed
            if show_duplicates:
                def on_continue():
                    # Force create client
                    pending = st.session_state.pending_client
                    try:
                        new_client = st.session_state.db.create_client(
                            ClientCreate(**pending), force=True
                        )
                        st.session_state.new_client_id = new_client.id
                        st.session_state.client_id = new_client.id
                        del st.session_state.duplicate_data
                        del st.session_state.pending_client
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

                def on_cancel():
                    del st.session_state.duplicate_data
                    del st.session_state.pending_client
                    st.rerun()

                show_duplicate_warning(duplicate_data, on_continue, on_cancel)

            # Show creation form if no client created yet and no duplicate warning
            elif not client_created:
                with st.form("new_client"):
                    name = st.text_input("Client Name *", placeholder="John Smith", key="new_name")
                    company = st.text_input("Company", placeholder="Acme Inc", key="new_company")
                    email = st.text_input("Email", placeholder="john@acme.com", key="new_email")

                    st.info("👆 Fill details and create client first, then process conversation")

                    col_check, col_submit = st.columns([1, 1])

                    with col_check:
                        check_dup = st.form_submit_button("🔍 Check Duplicates", use_container_width=True)

                    with col_submit:
                        submit = st.form_submit_button("➕ Create Client", use_container_width=True)

                    if check_dup and name:
                        duplicates = st.session_state.db.find_potential_duplicates(name, email, company)
                        if duplicates:
                            st.session_state.pending_client = {'name': name, 'company': company, 'email': email}
                            st.session_state.duplicate_data = duplicates
                            st.rerun()
                        else:
                            st.success("✅ No duplicates found!")

                    if submit and name:
                        try:
                            new_client = st.session_state.db.create_client(
                                ClientCreate(name=name, company=company, email=email)
                            )
                            st.session_state.new_client_id = new_client.id
                            st.session_state.client_id = new_client.id
                            st.success(f"✅ Created: {name}")
                            st.rerun()
                        except DuplicateClientError as e:
                            st.session_state.pending_client = {'name': name, 'company': company, 'email': email}
                            st.session_state.duplicate_data = e.args[1]
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

    with col2:
        st.subheader("Conversation Input")

        # Disable conversation input if no client selected (for new client flow)
        input_disabled = (client_mode == "New Client" and not client_id)

        if input_disabled:
            st.warning("⚠️ Please create the client first (left side) before entering conversation")

        conversation = st.text_area(
            "Paste conversation text",
            height=300,
            placeholder="""Example:
Me: Hi Sarah, thanks for taking the call today.
Sarah: No problem, I've been looking at your proposal...
...""",
            help="Paste the raw conversation text here. AI will extract structured data.",
            disabled=input_disabled,
            key="conversation_input"
        )

        col_btn1, col_btn2, _ = st.columns([1, 1, 2])

        with col_btn1:
            process_disabled = not client_id or not conversation or input_disabled
            process_btn = st.button(
                "🤖 Process with AI",
                type="primary",
                use_container_width=True,
                disabled=process_disabled
            )

        with col_btn2:
            if st.button("Clear", use_container_width=True):
                # Clear all session state
                for key in ['crm_data', 'followups', 'conversation', 'client_id',
                           'new_client_id', 'duplicate_data', 'pending_client',
                           'new_client_created', 'preselected_client_id']:
                    st.session_state.pop(key, None)
                st.rerun()

    # Processing section
    if process_btn and client_id and conversation:
        with st.spinner("🧠 AI analyzing conversation..."):
            try:
                # Get context for existing clients
                context = "New client"
                if client_mode == "Existing Client":
                    context = st.session_state.memory.get_context_for_ai(client_id)

                # Extract CRM data
                crm_data = extract_crm_data(conversation, context)
                st.session_state.crm_data = crm_data
                st.session_state.conversation = conversation
                st.session_state.client_id = client_id

                # Generate follow-ups
                with st.spinner("✍️ Generating follow-ups..."):
                    followups = generate_followups(client_id, crm_data.dict())
                    st.session_state.followups = followups

                st.success("✅ Analysis complete!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.info("Check that Ollama is running: `ollama serve`")

    # Display results
    if st.session_state.crm_data:
        st.markdown("---")
        st.subheader("📊 Extracted CRM Data")

        crm = st.session_state.crm_data

        cols = st.columns(3)
        with cols[0]:
            st.metric("Deal Stage", crm.deal_stage.replace("_", " ").title())
        with cols[1]:
            interest_class = crm.interest_level
            st.markdown(f"**Interest Level:** <span class='stage-badge {interest_class}'>{crm.interest_level.upper()}</span>", unsafe_allow_html=True)
        with cols[2]:
            if crm.followup_date:
                st.metric("Follow-up Date", crm.followup_date)

        with st.expander("View Details", expanded=True):
            st.write("**Summary:**")
            st.info(crm.summary)

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Next Action:**")
                st.success(crm.next_action)
            with col2:
                if crm.objections:
                    st.write("**Objections:**")
                    st.warning(crm.objections)

        # Show generated follow-ups
        if st.session_state.followups:
            st.subheader("📧 Generated Follow-ups")

            tabs = st.tabs(["📧 Email", "💬 WhatsApp Message"])

            with tabs[0]:
                st.text_area("Copy this email", st.session_state.followups.email_text, height=200, key="email_display")
                if st.button("📋 Copy Email to Clipboard", key="copy_email_btn"):
                    st.toast("✅ Email copied!")

            with tabs[1]:
                st.text_area("Copy this message", st.session_state.followups.message_text, height=100, key="msg_display")
                if st.button("📋 Copy Message to Clipboard", key="copy_msg_btn"):
                    st.toast("✅ Message copied!")

        # Save section
        st.markdown("---")
        col_save, col_discard = st.columns([1, 1])

        with col_save:
            if st.button("💾 Save to Database", type="primary", use_container_width=True):
                try:
                    # Save interaction first
                    interaction_data = InteractionCreate(
                        client_id=st.session_state.client_id,
                        raw_text=st.session_state.conversation,
                        summary=crm.summary,
                        deal_stage=crm.deal_stage.value,
                        objections=crm.objections,
                        interest_level=crm.interest_level.value,
                        next_action=crm.next_action,
                        followup_date=crm.followup_date
                    )

                    interaction = st.session_state.db.create_interaction(interaction_data)
                    st.success(f"✅ Interaction saved (ID: {interaction.id})")

                    # Save follow-ups if they exist
                    if st.session_state.followups:
                        followup = st.session_state.db.create_followup(
                            interaction.id,
                            st.session_state.followups.email_text,
                            st.session_state.followups.message_text
                        )
                        st.success(f"✅ Follow-ups saved (ID: {followup.id})")

                    st.balloons()

                    # Clear session state
                    for key in ['crm_data', 'followups', 'conversation', 'client_id',
                               'new_client_id', 'duplicate_data', 'pending_client',
                               'new_client_created']:
                        st.session_state.pop(key, None)

                    st.info("Redirecting to Follow-ups page...")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Save failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

        with col_discard:
            if st.button("🗑️ Discard", use_container_width=True):
                for key in ['crm_data', 'followups', 'conversation']:
                    st.session_state.pop(key, None)
                st.rerun()

def page_clients():
    """Page 2: Client list with detail view and deletion."""
    st.markdown('<div class="main-header">👥 Clients</div>', unsafe_allow_html=True)

    # Initialize view/delete states if not exists
    if 'view_client_id' not in st.session_state:
        st.session_state.view_client_id = None
    if 'delete_client_id' not in st.session_state:
        st.session_state.delete_client_id = None

    # Tabs for active and deleted clients
    tab1, tab2 = st.tabs(["Active Clients", "Deleted Clients"])

    with tab1:
        # Search
        search = st.text_input("🔍 Search clients", placeholder="Type name or company...", key="search_active")

        if search:
            clients = st.session_state.db.search_clients(search, include_inactive=False)
        else:
            clients = st.session_state.db.get_all_clients(include_inactive=False)

        if not clients:
            st.info("No active clients found.")
        else:
            st.write(f"Found {len(clients)} active clients")

            for idx, client in enumerate(clients):
                # Create a container for each client row
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

                    with col1:
                        st.write(f"**{client.name}**")
                        if client.company:
                            st.caption(client.company)

                    with col2:
                        if client.email:
                            st.caption(f"📧 {client.email}")

                    with col3:
                        # Stats
                        try:
                            stats = st.session_state.db.get_client_stats(client.id)
                            st.caption(f"📝 {stats['total_interactions']} interactions")
                        except:
                            st.caption("📝 0 interactions")

                    with col4:
                        view_col, del_col = st.columns(2)
                        with view_col:
                            if st.button("View", key=f"view_btn_{client.id}_{idx}", use_container_width=True):
                                st.session_state.view_client_id = client.id
                                st.session_state.delete_client_id = None  # Clear delete state
                                st.rerun()
                        with del_col:
                            if st.button("🗑️", key=f"del_btn_{client.id}_{idx}", use_container_width=True):
                                st.session_state.delete_client_id = client.id
                                st.session_state.view_client_id = None  # Clear view state
                                st.rerun()

                    # Show view details inline if selected
                    if st.session_state.view_client_id == client.id:
                        with st.expander(f"📋 Details for {client.name}", expanded=True):
                            col1, col2 = st.columns([3, 1])
                            with col2:
                                if st.button("❌ Close", key=f"close_view_{client.id}", use_container_width=True):
                                    st.session_state.view_client_id = None
                                    st.rerun()

                            # Client info editor
                            with st.form(f"edit_client_{client.id}"):
                                new_name = st.text_input("Name", client.name, key=f"edit_name_{client.id}")
                                new_company = st.text_input("Company", client.company or "", key=f"edit_comp_{client.id}")
                                new_email = st.text_input("Email", client.email or "", key=f"edit_email_{client.id}")

                                if st.form_submit_button("💾 Update Client", use_container_width=True):
                                    st.session_state.db.update_client(
                                        client.id,
                                        name=new_name,
                                        company=new_company,
                                        email=new_email
                                    )
                                    st.success("Updated!")
                                    st.rerun()

                            # Timeline
                            try:
                                timeline = st.session_state.memory.get_client_timeline(client.id)

                                if not timeline:
                                    st.info("No interactions yet.")
                                else:
                                    st.subheader("Interaction History")
                                    for item in timeline:
                                        with st.container():
                                            st.markdown(f"""
                                            <div style="border-left: 3px solid #0d6efd; padding-left: 1rem; margin-bottom: 1rem;">
                                                <small>{item['date'].strftime('%Y-%m-%d %H:%M')}</small><br>
                                                <strong>Stage:</strong> {item['stage'].replace('_', ' ').title()} | 
                                                <strong>Interest:</strong> {item['interest'].upper()}<br>
                                                <em>{item['summary']}</em><br>
                                                <small>Next: {item['next_action']}</small>
                                            </div>
                                            """, unsafe_allow_html=True)

                                            # Show follow-up if exists
                                            interactions = st.session_state.db.get_client_interactions(client.id)
                                            if interactions:
                                                followup = st.session_state.db.get_followup(interactions[-1].id)
                                                if followup:
                                                    with st.expander("View Follow-ups"):
                                                        st.write("**Email:**")
                                                        st.text_area("", followup.email_text, height=100, key=f"view_email_{item['date']}_{client.id}")
                                                        st.write("**Message:**")
                                                        st.text_area("", followup.message_text, height=60, key=f"view_msg_{item['date']}_{client.id}")

                            except Exception as e:
                                st.error(f"Error loading history: {str(e)}")

                        st.divider()

                    # Show delete confirmation inline if selected
                    if st.session_state.delete_client_id == client.id:
                        with st.container():
                            st.error(f"⚠️ Are you sure you want to delete {client.name}?")
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("✅ Yes, Delete", key=f"confirm_del_{client.id}", type="primary", use_container_width=True):
                                    st.session_state.db.delete_client(client.id, soft_delete=True)
                                    st.success(f"Deleted {client.name}")
                                    st.session_state.delete_client_id = None
                                    st.rerun()
                            with col2:
                                if st.button("❌ Cancel", key=f"cancel_del_{client.id}", use_container_width=True):
                                    st.session_state.delete_client_id = None
                                    st.rerun()

                        st.divider()
                    else:
                        st.divider()

    with tab2:
        st.info("View and restore deleted clients")

        # Get deleted clients (is_active = 0 or NULL)
        try:
            all_clients = st.session_state.db.get_all_clients(include_inactive=True)
            deleted_clients = [c for c in all_clients if hasattr(c, 'is_active') and c.is_active == 0]
        except Exception as e:
            st.error(f"Error loading deleted clients: {str(e)}")
            deleted_clients = []

        if not deleted_clients:
            st.write("No deleted clients.")
        else:
            st.write(f"Found {len(deleted_clients)} deleted client(s)")

            for idx, client in enumerate(deleted_clients):
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])

                    with col1:
                        st.write(f"**{client.name}** (Deleted)")
                        if client.company:
                            st.caption(f"🏢 {client.company}")
                        if client.email:
                            st.caption(f"📧 {client.email}")
                        # Show stats
                        try:
                            stats = st.session_state.db.get_client_stats(client.id)
                            st.caption(f"📝 {stats['total_interactions']} interactions | Last: {stats['last_contact'][:10] if stats['last_contact'] else 'N/A'}")
                        except:
                            pass

                    with col2:
                        if st.button("🔄 Restore", key=f"restore_{client.id}_{idx}", use_container_width=True):
                            st.session_state.db.restore_client(client.id)
                            st.success(f"Restored {client.name}")
                            st.rerun()

                    with col3:
                        # Permanent delete with confirmation
                        if st.button("💥 Delete Forever", key=f"perm_del_{client.id}_{idx}", use_container_width=True):
                            st.session_state.perm_delete_client_id = client.id
                            st.rerun()

                    # Permanent delete confirmation
                    if st.session_state.get('perm_delete_client_id') == client.id:
                        st.error("⚠️ WARNING: This will PERMANENTLY delete all data including interactions and follow-ups!")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("💀 YES, DELETE FOREVER", key=f"confirm_perm_{client.id}", type="primary", use_container_width=True):
                                st.session_state.db.delete_client(client.id, soft_delete=False)
                                st.success(f"Permanently deleted {client.name}")
                                st.session_state.perm_delete_client_id = None
                                st.rerun()
                        with col2:
                            if st.button("❌ Cancel", key=f"cancel_perm_{client.id}", use_container_width=True):
                                st.session_state.perm_delete_client_id = None
                                st.rerun()

                    st.divider()

def page_followups():
    """Page 3: Show active deals needing follow-up with quick actions."""
    st.markdown('<div class="main-header">📧 Follow-ups</div>', unsafe_allow_html=True)

    # SECTION 1: Recently Generated (from current session - not saved yet)
    if st.session_state.get('followups') and st.session_state.get('crm_data'):
        st.subheader("🆕 Just Generated (Save to store)")

        client = None
        if st.session_state.get('client_id'):
            client = st.session_state.db.get_client(st.session_state.client_id)

        crm = st.session_state.crm_data

        with st.container():
            col1, col2 = st.columns([1, 3])

            with col1:
                st.write(f"**{client.name if client else 'Current Client'}**")
                st.caption("Just now")
                interest_class = crm.interest_level
                st.markdown(f"**Interest:** <span style='background-color: #fd7e14; color: black; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.875rem; font-weight: 600;'>{crm.interest_level.upper()}</span>", unsafe_allow_html=True)
                st.caption(f"Stage: {crm.deal_stage.replace('_', ' ').title()}")
                if crm.followup_date:
                    st.caption(f"📅 Follow-up: {crm.followup_date}")

            with col2:
                tabs = st.tabs(["📧 Email", "💬 WhatsApp Message", "📝 Context"])

                with tabs[0]:
                    st.text_area("Copy this email", st.session_state.followups.email_text, height=200, key="new_email")
                    if st.button("📋 Copy Email", key="copy_new_email"):
                        st.toast("✅ Email copied to clipboard!")

                with tabs[1]:
                    st.text_area("Copy this message", st.session_state.followups.message_text, height=100, key="new_msg")
                    if st.button("📋 Copy Message", key="copy_new_msg"):
                        st.toast("✅ Message copied to clipboard!")

                with tabs[2]:
                    st.write(f"**Summary:** {crm.summary}")
                    st.write(f"**Next Action:** {crm.next_action}")
                    if crm.objections:
                        st.warning(f"**Objections:** {crm.objections}")

            st.info("💡 Go to 'Add Interaction' page and click 'Save to Database' to store this")
            st.divider()

    # SECTION 2: Active Deals Needing Follow-up (NOT closed won/lost)
    st.subheader("🎯 Active Deals Needing Follow-up")

    # Get all active clients with their latest interaction
    try:
        clients = st.session_state.db.get_all_clients(include_inactive=False)

        active_deals = []
        for client in clients:
            interactions = st.session_state.db.get_client_interactions(client.id)
            if interactions:
                latest = interactions[0]  # Most recent

                # Only show if NOT closed won/lost
                if latest.deal_stage not in ['closed_won', 'closed_lost']:
                    # Check if follow-up already generated for this interaction
                    existing_followup = st.session_state.db.get_followup(latest.id)

                    active_deals.append({
                        'client': client,
                        'interaction': latest,
                        'has_followup': existing_followup is not None,
                        'followup': existing_followup
                    })

        # Sort by follow-up date (urgent first)
        active_deals.sort(key=lambda x: x['interaction'].followup_date or '9999-12-31')

    except Exception as e:
        st.error(f"Error loading deals: {str(e)}")
        active_deals = []

    if not active_deals:
        st.info("No active deals needing follow-up. All caught up! 🎉")
    else:
        st.write(f"**{len(active_deals)} deals need attention**")

        # Filter options
        col1, col2 = st.columns([1, 1])
        with col1:
            stage_filter = st.multiselect(
                "Filter by stage",
                ["prospecting", "qualification", "proposal", "negotiation", "nurture"],
                default=[]
            )
        with col2:
            urgency_filter = st.selectbox(
                "Filter by urgency",
                ["All", "Overdue", "Today", "This Week", "No Date"],
                index=0
            )

        today = date.today()

        for deal in active_deals:
            client = deal['client']
            inter = deal['interaction']

            # Apply filters
            if stage_filter and inter.deal_stage not in stage_filter:
                continue

            # Urgency filter
            followup_date = None
            if inter.followup_date:
                try:
                    followup_date = datetime.strptime(inter.followup_date, "%Y-%m-%d").date()
                except:
                    pass

            if urgency_filter == "Overdue" and (not followup_date or followup_date >= today):
                continue
            if urgency_filter == "Today" and followup_date != today:
                continue
            if urgency_filter == "This Week":
                if not followup_date or followup_date < today or followup_date > today + timedelta(days=7):
                    continue
            if urgency_filter == "No Date" and followup_date:
                continue

            # Determine urgency color
            if followup_date:
                if followup_date < today:
                    urgency_emoji = "🔴"
                    urgency_text = "OVERDUE"
                    urgency_color = "#dc3545"
                elif followup_date == today:
                    urgency_emoji = "🟡"
                    urgency_text = "TODAY"
                    urgency_color = "#fd7e14"
                else:
                    days_until = (followup_date - today).days
                    urgency_emoji = "🟢"
                    urgency_text = f"{days_until} days"
                    urgency_color = "#28a745"
            else:
                urgency_emoji = "⚪"
                urgency_text = "No date set"
                urgency_color = "#6c757d"

            with st.container():
                # Header row
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                with col1:
                    st.write(f"**{client.name}**")
                    if client.company:
                        st.caption(f"🏢 {client.company}")
                    if client.email:
                        st.caption(f"📧 {client.email}")

                with col2:
                    st.caption(f"Stage: **{inter.deal_stage.replace('_', ' ').title()}**")
                    interest_color = "#dc3545" if inter.interest_level == "hot" else "#fd7e14" if inter.interest_level == "warm" else "#17a2b8" if inter.interest_level == "cold" else "#6c757d"
                    st.markdown(f"<span style='background-color: {interest_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.875rem; font-weight: 600;'>{inter.interest_level.upper()}</span>", unsafe_allow_html=True)

                with col3:
                    st.caption(f"Last contact: {inter.date.strftime('%Y-%m-%d')}")
                    st.markdown(f"<span style='color:{urgency_color};font-weight:bold;'>{urgency_emoji} {urgency_text}</span>", unsafe_allow_html=True)
                    if inter.followup_date:
                        st.caption(f"Follow-up: {inter.followup_date}")

                with col4:
                    # New Interaction button - redirects to Add Interaction with pre-selected client
                    if st.button("📝 New Interaction", key=f"new_int_{client.id}", use_container_width=True):
                        st.session_state.preselected_client_id = client.id
                        st.session_state.page = "📝 Add Interaction"
                        st.rerun()

                # Expandable details
                with st.expander(f"💬 {inter.summary[:60]}...", expanded=False):
                    st.write(f"**Full Summary:** {inter.summary}")
                    st.write(f"**Next Action Required:** {inter.next_action}")
                    if inter.objections:
                        st.warning(f"**Objections:** {inter.objections}")

                    # If follow-up already generated, show it
                    if deal['has_followup'] and deal['followup']:
                        st.success("✅ Follow-up content already generated")

                        tabs = st.tabs(["📧 Email", "💬 WhatsApp"])

                        with tabs[0]:
                            st.text_area("Email content", deal['followup'].email_text, height=150, key=f"email_{inter.id}")
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                if st.button("📋 Copy Email", key=f"copy_email_{inter.id}"):
                                    st.toast("✅ Email copied!")
                            with col2:
                                if st.button("✉️ Send Email", key=f"send_email_{inter.id}"):
                                    st.info("Opening email client... (simulation)")

                        with tabs[1]:
                            st.text_area("Message content", deal['followup'].message_text, height=80, key=f"msg_{inter.id}")
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                if st.button("📋 Copy Message", key=f"copy_msg_{inter.id}"):
                                    st.toast("✅ Message copied!")
                            with col2:
                                if st.button("💬 Send WhatsApp", key=f"send_msg_{inter.id}"):
                                    st.info("Opening WhatsApp... (simulation)")

                        # CLOSE FOLLOW-UP OPTIONS - NOW VISIBLE
                        st.markdown("---")
                        st.write("**Close Follow-up:**")

                        close_cols = st.columns(3)

                        with close_cols[0]:
                            if st.button("✅ Mark as Won", key=f"won_{inter.id}", type="primary", use_container_width=True):
                                # Create new interaction marking as won
                                won_data = InteractionCreate(
                                    client_id=client.id,
                                    raw_text="Deal marked as closed won from follow-up page",
                                    summary=f"Deal closed successfully. Previous: {inter.summary}",
                                    deal_stage="closed_won",
                                    objections=None,
                                    interest_level="hot",
                                    next_action="None - Deal closed",
                                    followup_date=None
                                )
                                st.session_state.db.create_interaction(won_data)
                                st.success("🎉 Marked as WON!")
                                st.rerun()

                        with close_cols[1]:
                            if st.button("❌ Mark as Lost", key=f"lost_{inter.id}", use_container_width=True):
                                lost_data = InteractionCreate(
                                    client_id=client.id,
                                    raw_text="Deal marked as closed lost from follow-up page",
                                    summary=f"Deal lost. Previous: {inter.summary}",
                                    deal_stage="closed_lost",
                                    objections=None,
                                    interest_level="cold",
                                    next_action="None - Deal lost",
                                    followup_date=None
                                )
                                st.session_state.db.create_interaction(lost_data)
                                st.warning("Marked as LOST")
                                st.rerun()

                        with close_cols[2]:
                            if st.button("⏭️ Skip/Defer", key=f"skip_{inter.id}", use_container_width=True):
                                skip_data = InteractionCreate(
                                    client_id=client.id,
                                    raw_text="Follow-up deferred",
                                    summary=f"Follow-up deferred. Previous: {inter.summary}",
                                    deal_stage=inter.deal_stage,
                                    objections=None,
                                    interest_level=inter.interest_level,
                                    next_action="Follow-up deferred by user",
                                    followup_date=None
                                )
                                st.session_state.db.create_interaction(skip_data)
                                st.info("⏭️ Follow-up deferred")
                                st.rerun()

                    else:
                        st.warning("⚠️ No follow-up content generated yet")

                        if st.button("✨ Generate Follow-up Now", key=f"gen_{inter.id}", type="primary"):
                            with st.spinner("Generating..."):
                                try:
                                    # Prepare data with proper None handling
                                    crm_data_dict = {
                                        'summary': inter.summary or '',
                                        'deal_stage': inter.deal_stage or 'prospecting',
                                        'interest_level': inter.interest_level or 'neutral',
                                        'next_action': inter.next_action or 'Follow up',
                                        'objections': inter.objections if inter.objections else None
                                    }

                                    followups = generate_followups(client.id, crm_data_dict)

                                    # Save to database
                                    st.session_state.db.create_followup(inter.id, followups.email_text, followups.message_text)
                                    st.success("✅ Generated and saved!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                                    import traceback
                                    st.code(traceback.format_exc())

                st.divider()

    # SECTION 3: All Saved Follow-ups (Archive)
    with st.expander("📚 View All Saved Follow-ups (Archive)"):
        try:
            all_followups = st.session_state.db.get_all_followups(include_inactive=False)
        except:
            all_followups = []

        if not all_followups:
            st.info("No saved follow-ups in archive.")
        else:
            st.write(f"Total archived: {len(all_followups)}")

            for followup, interaction, client in all_followups[:10]:  # Show last 10
                st.write(f"**{client.name}** - {interaction.date.strftime('%Y-%m-%d')}")
                st.caption(f"Stage: {interaction.deal_stage} | Interest: {interaction.interest_level}")
                with st.expander("View content"):
                    st.text_area("Email", followup.email_text, height=100, key=f"arch_email_{followup.id}")
                    st.text_area("Message", followup.message_text, height=60, key=f"arch_msg_{followup.id}")
                st.divider()


def page_dashboard():
    """Page 4: Analytics dashboard."""
    st.markdown('<div class="main-header">📊 Dashboard</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    # Stats
    clients = st.session_state.db.get_all_clients(include_inactive=False)
    all_clients = st.session_state.db.get_all_clients(include_inactive=True)  # Fixed: removed double "all_"
    deleted_count = len(all_clients) - len(clients)
    interactions = st.session_state.db.get_recent_interactions(1000)

    with col1:
        st.metric("Active Clients", len(clients))
    with col2:
        st.metric("Deleted Clients", deleted_count)
    with col3:
        st.metric("Total Interactions", len(interactions))
    with col4:
        # Pipeline value (simulated)
        pipeline_stats = st.session_state.db.get_pipeline_stats(include_inactive=False)
        active_deals = sum(count for stage, count in pipeline_stats.items()
                           if stage not in ['closed_won', 'closed_lost'])
        st.metric("Active Deals", active_deals)

    # Pipeline chart
    st.subheader("Pipeline Overview")
    if pipeline_stats:
        df = pd.DataFrame([
            {"Stage": stage.replace("_", " ").title(), "Count": count}
            for stage, count in pipeline_stats.items()
        ])
        st.bar_chart(df.set_index("Stage"))

    # Recent activity
    st.subheader("Recent Activity")
    recent = st.session_state.db.get_recent_interactions(5)
    for interaction, client in recent:
        st.write(f"**{client.name}** - {interaction.deal_stage.replace('_', ' ').title()} "
                 f"({interaction.date.strftime('%Y-%m-%d')})")


def main():
    """Main app entry."""
    page = render_sidebar()

    if "Add Interaction" in page:
        page_add_interaction()
    elif "Clients" in page:
        page_clients()
    elif "Follow-ups" in page:
        page_followups()
    else:
        page_dashboard()

if __name__ == "__main__":
    main()