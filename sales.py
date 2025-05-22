import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import hashlib
import joblib
from datetime import datetime
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_percentage_error, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Dashboard test", page_icon=":bar_chart:", layout="wide")

st.markdown('<style>{}</style>'.format(open('style.css').read()), unsafe_allow_html=True)


users = {
    "admin": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin"
    },
    "david": {
        "password": hashlib.sha256("sales2025".encode()).hexdigest(),
        "sales_member": "David Lee"
    },
    "james": {
        "password": hashlib.sha256("sales2025".encode()).hexdigest(),
        "sales_member": "James Smith"
    },
    "maria": {
        "password": hashlib.sha256("sales2025".encode()).hexdigest(),
        "sales_member": "Maria Garcia"
    },
    "michael": {
        "password": hashlib.sha256("sales2025".encode()).hexdigest(),
        "sales_member": "Michael Brown"
    },
    "sarah": {
        "password": hashlib.sha256("sales2025".encode()).hexdigest(),
        "sales_member": "Sarah Johnson"
    }
}

# Add these right after your imports
def create_features(df):
    """Create time-based features from timestamp"""
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Cyclical encoding for hour and dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/23)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/23)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/6)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/6)
    
    return df

def add_lag_features(df, target_col='visits', lags=[1, 24, 168]):
    """Add lag and rolling features for time series forecasting"""
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling features
    df['rolling_mean_24'] = df[target_col].rolling(window=24).mean()
    df['rolling_mean_168'] = df[target_col].rolling(window=168).mean()
    
    return df

# Authentication function
def login(username, password):
    user = users.get(username)
    if user and user["password"] == hashlib.sha256(password.encode()).hexdigest():
        return user
    return None

# Session state setup
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

if not st.session_state.logged_in:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = login(username, password)
        if user:
            st.success("Login successful!")
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.user_info = user
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.stop()


if 'targets' not in st.session_state:
    st.session_state.targets = {
        'team_monthly_target': 100000,  # Default team target
        'user_targets': {
            "David Lee": 25000,
            "James Smith": 30000,
            "Maria Garcia": 35000,
            "Michael Brown": 40000,
            "Sarah Johnson": 45000
        },
        'product_targets': {
            # Add your product targets here
            "AI Virtual Assistant": 40000,
            "AI Chatbot": 35000,
            "Smart Scheduling Bot": 30000,
            "Sales Dashboard": 25000
        }
    }

with st.sidebar:
    if st.session_state.get("logged_in", False):
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()


# Pre-load data with caching
@st.cache_data
def load_data():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to CSV
    file_path = os.path.join(script_dir, "pd11_data.csv")
    
    # Load CSV
    df = pd.read_csv(file_path)
    
    # Parse datetime and add features
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date
    df['day_name'] = df['timestamp'].dt.day_name()
    df['week'] = df['timestamp'].dt.to_period('W').astype(str)
    df['month'] = df['timestamp'].dt.to_period('M').astype(str)
    
    return df


# Load data once
df = load_data()

@st.cache_resource
def load_models():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load model files from local directory
    model_artifacts = joblib.load(os.path.join(script_dir, "model_artifacts.pkl"))
    traffic_model = joblib.load(os.path.join(script_dir, "traffic_model.pkl"))
    engagement_model = joblib.load(os.path.join(script_dir, "engagement_model.pkl"))
    engagement_scaler = joblib.load(os.path.join(script_dir, "engagement_scaler.pkl"))
    growth_model = joblib.load(os.path.join(script_dir, "growth_model.pkl"))
    page_encoder = joblib.load(os.path.join(script_dir, "page_encoder.pkl"))

    return {
        "artifacts": model_artifacts,
        "traffic_model": traffic_model,
        "engagement_model": engagement_model,
        "engagement_scaler": engagement_scaler,
        "growth_model": growth_model,
        "page_encoder": page_encoder
    }


def create_gauge(current_value, target_value, label, height=200):  # Added height parameter with default
    if target_value == 0:
        return None
    
    max_val = max(current_value, target_value) * 1.2
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_value,
        delta={"reference": target_value, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
        title={"text": label},
        gauge={
            "axis": {"range": [0, max_val]},
            "bar": {"color": "seagreen"},
            "steps": [
                {"range": [0, target_value], "color": "lightgray"},
                {"range": [target_value, max_val], "color": "palegreen"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": target_value
            }
        }
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=height  # Now uses parameter
    )
    return fig


# Title and styling
#st.markdown('<style>div.block-container{padding-top:0rem !important;padding-bottom: 1rem;} header, footer {visibility: hidden;} </style>', unsafe_allow_html=True)

# Sidebar Filters - optimized to run only once
with st.sidebar:
    # Title in sidebar
    st.markdown("# :bar_chart: LogInSight")
    # Filters section
    st.header("🔎 Filter Data")
    
    # Sales member selection
    sales_team = ['All'] + sorted([m for m in df['sales_member'].unique() if m not in [None, 'Not Commissionable']])
    user_default = st.session_state.user_info.get("sales_member", 'All')
    default_index = sales_team.index(user_default) if user_default in sales_team else 0
    selected_sales_member = st.selectbox("Select Sales Member", options=sales_team, index=default_index)

    # Combined month and day picker
    st.subheader("📅 Date Selection")
    
    # Get min and max dates from your data
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    
    # Default to the range of the last 30 days
    default_end_date = datetime.now().date()
    default_start_date = default_end_date - pd.Timedelta(days=30)
    if default_start_date < min_date:
        default_start_date = min_date
    if default_end_date > max_date:
        default_end_date = max_date
    
    # Create a date input that allows range selection
    selected_dates = st.date_input(
        "Select date or range",
        value=(default_start_date, default_end_date),  # Default to 1st of current month
        min_value=min_date,
        max_value=max_date,
        format="YYYY/MM/DD"
    )
    
    # Determine if it's a range or single date
    if isinstance(selected_dates, tuple):
        if len(selected_dates) == 2:
            start_date, end_date = selected_dates
        else:
            start_date = end_date = selected_dates[0]
    else:
        start_date = end_date = selected_dates
    
    # Country selection
    all_countries = df['country'].unique()
    countries = st.multiselect("Select Country", options=all_countries, default=all_countries)

# Update the filter_data function
@st.cache_data
def filter_data(df, countries, selected_sales_member, start_date, end_date):
    mask = df['country'].isin(countries)
    
    # Convert dates to datetime if they aren't already
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # Include entire end date
    
    # Filter by date range
    mask &= (df['timestamp'] >= start_date) & (df['timestamp'] < end_date)
    
    if selected_sales_member != 'All':
        mask &= (df['sales_member'] == selected_sales_member)
    return df[mask].copy()

# Filter the data
df_filtered = filter_data(df, countries, selected_sales_member, start_date, end_date)


# Pre-compute common aggregations
@st.cache_data
def compute_common_metrics(_df):
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    _df['day_name'] = pd.Categorical(_df['day_name'], categories=day_order, ordered=True)
    
    metrics = {
        'hourly_traffic': _df['hour'].value_counts().sort_index(),
        'daily_traffic': _df['day_name'].value_counts().sort_index(),
        'trend_data': _df.groupby('date').size().reset_index(name='visits'),
        'top_countries': _df['country'].value_counts().nlargest(10).reset_index(),
        'top_actions': _df['action'].value_counts().nlargest(10).reset_index(),
        'conversion_by_action': _df.groupby('action')['converted'].mean().reset_index(),
        'device_counts': _df['device_type'].value_counts().reset_index(),
        'visitor_conversion': _df.groupby('is_new_visitor')['converted'].mean().reset_index()
    }
    return metrics

metrics = compute_common_metrics(df_filtered)

team_performance = df_filtered[(df_filtered['sales_member'].notna()) & 
                             (df_filtered['sales_member'] != 'Not Commissionable')]

@st.cache_data
def compute_team_metrics(_df, df_shape):
    all_countries = pd.DataFrame({
        'country': pd.unique(_df['country']),
        'exists': True
    })
    
    country_revenue = _df.groupby('country')['deal_size'].sum().reset_index()
    country_revenue = pd.merge(all_countries, country_revenue, 
                              on='country', how='left')
    country_revenue['deal_size'] = country_revenue['deal_size'].fillna(0)
    
    return {
        'total_revenue': _df['deal_size'].sum(),
        'avg_deal': _df.loc[_df['deal_size'] > 0, 'deal_size'].mean(),
        'conv_rate': _df['converted'].mean(),
        'revenue_by_member': _df.groupby('sales_member')['deal_size'].sum().sort_values(),
        'country_revenue': country_revenue,
        'product_rev': _df.groupby('product_name')['deal_size'].sum().nlargest(5),
        'product_conv': _df.groupby('product_name')['converted'].mean().nlargest(5)
    }

team_metrics = compute_team_metrics(team_performance, team_performance.shape) if not team_performance.empty else None
team_target = st.session_state.targets['team_monthly_target']

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🛍️ Sales Overview", "🛍️ Performance Analysis", "🧭 User Behavior",  "🔮 AI Predictions"])

with tab3:
    # SECTION 1: TRAFFIC PATTERNS
    st.subheader("When Do Visitors Engage? Key Traffic Patterns")
    
    # Insight header before the charts
    st.caption("""
    🕒 **Peak Hours Analysis**: Our data reveals when users are most active - crucial for timing campaigns and server scaling.
    """)
    
    col1, col2 = st.columns([1.2, 1.2])
    
    with col1:
        hourly_traffic = df_filtered['hour'].value_counts().sort_index()
        peak_hour = hourly_traffic.idxmax()
        
        fig = px.line(
            x=hourly_traffic.index, 
            y=hourly_traffic.values,
            title=f'<b>Traffic Peaks at {peak_hour}:00</b> – {hourly_traffic.max()} visits',
            labels={'x': 'Hour of Day', 'y': 'Visits'},
            markers=True,
            line_shape='spline'
        )
        fig.update_traces(
            line_color='#2a9d8f', 
            line_width=3,
            hovertemplate="<b>%{x}:00</b><br>%{y} visits<extra></extra>"
        )
        fig.add_hrect(
            y0=hourly_traffic.mean(), y1=hourly_traffic.max(),
            fillcolor="#264653", opacity=0.2,
            annotation_text=f"Peak hours: {peak_hour-2}:00-{peak_hour+2}:00", 
            annotation_position="top left"
        )
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=60, b=0),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_traffic = df_filtered['day_name'].value_counts().reindex(day_order, fill_value=0)
        weekday_avg = daily_traffic[:5].mean()
        weekend_diff = int((daily_traffic[-2:].mean() - weekday_avg) / weekday_avg * 100)
        
        fig = px.bar(
            x=daily_traffic.index, 
            y=daily_traffic.values,
            title=f'<b>Weekend Traffic: {weekend_diff}% {"Higher" if weekend_diff > 0 else "Lower"}</b> vs Weekdays',
            labels={'x': '', 'y': 'Visits'},
            color=[1 if day in ['Saturday','Sunday'] else 0 for day in daily_traffic.index],
            color_continuous_scale=['#2a9d8f','#e9c46a']
        )
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>%{y} visits<extra></extra>",
            marker_line_width=0
        )
        fig.update_layout(
            height=200,
            showlegend=False,
            margin=dict(l=0, r=0, t=60, b=0),
            coloraxis_showscale=False,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

    # SECTION 2: USER BEHAVIOR
    st.subheader("Who Converts Best? Engagement Insights")
    st.caption("""
    👥 **Visitor Segmentation**: New vs returning visitor analysis reveals where to focus acquisition vs retention efforts.
    """)
    
    col3, col4 = st.columns([1.2, 1.2])
    
    with col3:
        visitor_data = metrics['visitor_conversion'].copy()
        visitor_data['is_new_visitor'] = visitor_data['is_new_visitor'].map({True: 'New', False: 'Returning'})
        conv_gap = int((visitor_data.loc[1,'converted'] - visitor_data.loc[0,'converted']) / visitor_data.loc[0,'converted'] * 100)
        
        fig = px.bar(
            visitor_data, 
            x='is_new_visitor', 
            y='converted',
            title=f'<b>Returning Visitors Convert {conv_gap}% Better</b>',
            labels={'converted': 'Conversion Rate %'},
            color='is_new_visitor',
            color_discrete_map={'New':'#e76f51','Returning':'#2a9d8f'}
        )
        fig.add_hline(
            y=visitor_data['converted'].mean(), 
            line_dash="dot",
            annotation_text="Average", 
            line_color="white"
        )
        fig.update_layout(
            height=200,
            showlegend=False,
            margin=dict(l=0, r=0, t=60, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        top_actions = metrics['top_actions'].sort_values('count', ascending=True)
        primary_action = top_actions.iloc[-1]['action']
        
        fig = px.bar(
            top_actions, 
            x='count', 
            y='action', 
            orientation='h',
            title=f'<b>Most Common Action:</b> {primary_action}',
            color='count',
            color_continuous_scale='Teal'
        )
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>%{x} users<extra></extra>",
            marker_line_width=0
        )
        fig.update_layout(
            height=200,
            showlegend=False,
            margin=dict(l=0, r=0, t=60, b=0),
            coloraxis_showscale=False,
            yaxis_autorange='reversed'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # SUMMARY INSIGHTS
    st.divider()
    st.subheader("Key Takeaways")
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("""
        🚀 **Opportunities**:
        - Boost conversions during peak traffic hours (2-4PM)
        - Weekend promotions could capitalize on higher engagement
        - Returning visitors are our best customers - strengthen retention
        """)
    
    with col6:
        st.markdown("""
        🔍 **Next Steps**:
        - A/B test campaigns during high-traffic/low-conversion windows
        - Develop loyalty program for returning visitors
        - Optimize the 'Add to Cart' user journey (top action)
        """)

with tab1:  # Team Performance
    # Calculate comparison period based on selected date range
    comparison_days = (end_date - start_date).days
    comparison_start = start_date - pd.Timedelta(days=comparison_days)
    comparison_end = start_date
    df_comparison = filter_data(df, countries, selected_sales_member, comparison_start, comparison_end)

    st.subheader(f"\U0001F3C6 {'Team' if selected_sales_member == 'All' else selected_sales_member} Performance Dashboard")

    # --- Monthly Target Summary ---
    if selected_sales_member == 'All':
        monthly_target = st.session_state.targets['team_monthly_target']
        achieved = team_metrics['total_revenue']
    else:
        monthly_target = st.session_state.targets['user_targets'].get(selected_sales_member, 0)
        user_df = team_performance[team_performance['sales_member'] == selected_sales_member]
        achieved = user_df['deal_size'].sum()

    achievement_pct = achieved / monthly_target if monthly_target > 0 else 0

    if achievement_pct < 0.7:
        color = 'red'
    elif achievement_pct < 1:
        color = 'orange'
    else:
        color = 'green'

    st.markdown(f"""
    <div style='padding:10px; border-radius:10px; background-color:{color}; color:white; text-align:center'>
        <b>{'Team' if selected_sales_member == 'All' else selected_sales_member} Monthly Target:</b><br>
        Achieved ${achieved:,.0f} / ${monthly_target:,.0f} ({achievement_pct:.1%})
    </div>
    """, unsafe_allow_html=True)

    # --- Top Metrics Row ---
    metric_cols = st.columns(4)
    current_revenue = team_metrics['total_revenue']
    current_conv_rate = team_metrics['conv_rate']
    current_avg_deal = team_metrics['avg_deal']
    current_deals = len(team_performance[team_performance['converted']])

    comparison_revenue = df_comparison['deal_size'].sum()
    comparison_conv_rate = df_comparison['converted'].mean()
    comparison_avg_deal = df_comparison.loc[df_comparison['deal_size'] > 0, 'deal_size'].mean()
    comparison_deals = len(df_comparison[df_comparison['converted']])

    with metric_cols[0]:
        target = st.session_state.targets['team_monthly_target'] 
        delta_target = (current_revenue - target)/target if target > 0 else 0
        st.metric("Revenue", f"${current_revenue:,.0f}", delta=f"{achievement_pct:.1%}", delta_color="normal")

    with metric_cols[1]:
        delta_conv = current_conv_rate - comparison_conv_rate
        st.metric("Conversion Rate", f"{current_conv_rate:.1%}", delta=f"{delta_conv:.1%}", delta_color="normal")

    with metric_cols[2]:
        delta_avg = current_avg_deal - comparison_avg_deal if not np.isnan(current_avg_deal) else 0
        st.metric("Avg Deal Size", f"${current_avg_deal:,.0f}" if not np.isnan(current_avg_deal) else "$0", delta=f"${delta_avg:,.0f}" if not np.isnan(delta_avg) else "$0")

    with metric_cols[3]:
        delta_deals = current_deals - comparison_deals
        st.metric("Deals Closed", current_deals, delta=delta_deals)

    col1, col2 = st.columns([1.4, 1])

    with col1:
        product_data = []
        product_targets = st.session_state.targets['product_targets']

        for product, target in product_targets.items():
            current = team_performance[team_performance['product_name'] == product]['deal_size'].sum()
            previous = df_comparison[df_comparison['product_name'] == product]['deal_size'].sum()

            product_data.append({
                'Product': product,
                'Current Revenue': current,
                'Previous Revenue': previous,
                'Target': target * ((end_date - start_date).days / 30),
                'Growth': (current - previous)/previous if previous > 0 else 0
            })

        df_products = pd.DataFrame(product_data).sort_values('Current Revenue', ascending=False)

        fig = px.bar(df_products, x='Product', y=['Current Revenue', 'Target'], title="Product Revenue vs Targets",
                     barmode='group', color_discrete_sequence=['#2ecc71', '#3498db'], labels={'value': 'Revenue ($)'})

        for i, row in df_products.iterrows():
            fig.add_annotation(x=row['Product'], y=max(row['Current Revenue'], row['Target']) * 1.05,
                               text=f"▲{row['Growth']:.0%}" if row['Growth'] > 0 else f"▼{abs(row['Growth']):.0%}",
                               showarrow=False,
                               font=dict(color='green' if row['Growth'] > 0 else 'red'))

        fig.update_layout(height=260)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Global Revenue Distribution**")
        fig = px.choropleth(team_metrics['country_revenue'], locations='country', locationmode='country names',
                            color='deal_size', hover_name='country', color_continuous_scale='Viridis',
                            labels={'deal_size': 'Revenue ($)'})
        fig.update_geos(showcountries=True, countrycolor="lightgray")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), geo=dict(projection_scale=1.1))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col2:
        fig = create_gauge(current_revenue, target, "Revenue Target", height=300)
        st.plotly_chart(fig, use_container_width=True)

        ranking_filtered = filter_data(df, countries=countries, selected_sales_member='All', start_date=start_date, end_date=end_date)

        ranking_data = []
        for member in st.session_state.targets['user_targets'].keys():
            member_data = ranking_filtered[ranking_filtered['sales_member'] == member]
            member_rev = member_data['deal_size'].sum()

            member_comparison_data = df_comparison[df_comparison['sales_member'] == member]
            prev_rev = member_comparison_data['deal_size'].sum()

            member_target = st.session_state.targets['user_targets'][member] * ((end_date - start_date).days / 30)
            member_deals = len(member_data[member_data['converted']])
            avg_deal = member_data['deal_size'].mean() if not member_data.empty else 0
            conv_rate = member_data['converted'].mean() if not member_data.empty else 0

            ranking_data.append({
                'Member': member,
                'Revenue': member_rev,
                'Target': member_target,
                'Achievement': (member_rev / member_target) if member_target > 0 else 0,
                'Deals': member_deals,
                'Avg Deal': avg_deal,
                'Conv Rate': conv_rate
            })

        df_ranking = pd.DataFrame(ranking_data)
        df_ranking['Rank'] = df_ranking['Revenue'].rank(ascending=False, method='min').astype(int)
        df_ranking = df_ranking.sort_values('Rank')

        display_df = df_ranking.copy()
        display_df['Revenue'] = display_df['Revenue'].apply(lambda x: f"${x:,.0f}")
        display_df['Target'] = display_df['Target'].apply(lambda x: f"${x:,.0f}")
        display_df['Avg Deal'] = display_df['Avg Deal'].apply(lambda x: f"${x:,.0f}" if not np.isnan(x) else "$0")
        display_df['Achievement'] = display_df['Achievement'].apply(lambda x: f"{x:.1%}")
        display_df['Conv Rate'] = display_df['Conv Rate'].apply(lambda x: f"{x:.1%}")

        achievement_values = display_df['Achievement'].str.rstrip('%').astype(float)/100
        colors = ['#ff7675' if a < 0.7 else '#fdcb6e' if a < 1 else '#55efc4' for a in achievement_values]

        st.markdown("**\U0001F3C6 Team Member Ranking**")
        fig = go.Figure(data=[go.Table(
            columnwidth=[0.8, 2, 1.2, 1.2, 1.2, 1, 1.2, 1.2],
            header=dict(
                values=["<b>Rank</b>", "<b>Member</b>", "<b>Revenue</b>", "<b>Target</b>", "<b>Achievement</b>", "<b>Deals</b>", "<b>Avg Deal</b>", "<b>Conv Rate</b>"],
                fill_color='#2c3e50',
                font=dict(color='white', size=11),
                align='center'
            ),
            cells=dict(
                values=[display_df.Rank, display_df.Member, display_df.Revenue, display_df.Target, display_df.Achievement, display_df.Deals, display_df['Avg Deal'], display_df['Conv Rate']],
                fill=dict(color=[
                    ['white']*len(display_df),
                    ['white']*len(display_df),
                    ['white']*len(display_df),
                    ['white']*len(display_df),
                    colors,
                    ['white']*len(display_df),
                    ['white']*len(display_df),
                    ['white']*len(display_df)
                ]),
                font=dict(size=11),
                align=['center', 'left', 'right', 'right', 'center', 'center', 'right', 'center']
            )
        )])

        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with tab2:  # Sales Insights
    # Calculate comparison period
    comparison_days = (end_date - start_date).days
    comparison_start = start_date - pd.Timedelta(days=comparison_days)
    comparison_end = start_date

    df_current = filter_data(df, countries, selected_sales_member, start_date, end_date)
    df_comparison = filter_data(df, countries, selected_sales_member, comparison_start, comparison_end)

    # --- Compact Metrics Section ---
    st.subheader("📊 Key Metrics")
    metric_cols = st.columns(4)

    # Current metrics
    current_revenue = team_metrics['total_revenue']
    current_conversions = len(df_current[df_current['converted']])
    current_conv_rate = team_metrics['conv_rate']
    current_avg_deal = df_current.loc[df_current['deal_size'] > 0, 'deal_size'].mean() if not df_current[df_current['deal_size'] > 0].empty else 0.0

    # Previous metrics
    comparison_revenue = df_comparison['deal_size'].sum()
    prev_conversions = len(df_comparison[df_comparison['converted']])
    comparison_conv_rate = df_comparison['converted'].mean()
    prev_avg_deal = df_comparison.loc[df_comparison['deal_size'] > 0, 'deal_size'].mean() if not df_comparison[df_comparison['deal_size'] > 0].empty else 0.0

    with metric_cols[0]:
        target = st.session_state.targets['team_monthly_target'] 
        delta_target = (current_revenue - target)/target if target > 0 else 0
        st.metric("Revenue", f"${current_revenue:,.0f}", delta=f"{achievement_pct:.1%}", delta_color="normal")

    with metric_cols[1]:
        delta_conv = current_conversions - prev_conversions
        st.metric("Conversions", current_conversions,
                  delta=int(delta_conv))

    with metric_cols[2]:
        delta_conv = current_conv_rate - comparison_conv_rate
        st.metric("Conversion Rate", f"{current_conv_rate:.1%}", delta=f"{delta_conv:.1%}", delta_color="normal")

    with metric_cols[3]:
        delta_avg = current_avg_deal - prev_avg_deal
        st.metric("Avg Deal", f"${current_avg_deal:,.0f}" if not np.isnan(current_avg_deal) else "$0",
                  delta=f"${delta_avg:,.0f}" if not np.isnan(delta_avg) else "$0")

    # --- Combined Visualization Section ---
    viz_cols = st.columns([1, 1.2])  # Right column slightly wider

    with viz_cols[0]:
        # Mini Revenue Trend
        st.markdown("**📈 Revenue Trend**")
        daily_rev = df_current.resample('D', on='timestamp')['deal_size'].sum().reset_index()
        fig = px.area(daily_rev, x='timestamp', y='deal_size',
                      height=200,
                      labels={'deal_size': ''})
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Mini Product Conversion
        st.markdown("**📊 Product Conversion**")
        current_prod_conv = df_current.groupby('product_name')['converted'].mean().nlargest(3)
        fig = px.bar(current_prod_conv * 100, orientation='h',
                     height=200,
                     labels={'value': 'Conversion %'})
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with viz_cols[1]:
        # Product Target Gauges
        st.markdown("**🎯 Product Targets**")
        product_targets = st.session_state.targets['product_targets']
        gauge_cols = st.columns(len(product_targets))

        for i, (product, target) in enumerate(product_targets.items()):
            with gauge_cols[i]:
                current = df_current[df_current['product_name'] == product]['deal_size'].sum()
                adjusted_target = target * ((end_date - start_date).days / 30)
                fig = create_gauge(current, adjusted_target, product[:12], height=180)  # Truncate long names
                if fig:
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Product Revenue Comparison
        st.markdown("**📦 Top Products**")
        current_prod_rev = df_current.groupby('product_name')['deal_size'].sum().nlargest(3)
        prev_prod_rev = df_comparison.groupby('product_name')['deal_size'].sum()

        comparison_data = pd.DataFrame({
            'product': current_prod_rev.index,
            'current': current_prod_rev.values,
            'previous': [prev_prod_rev.get(p, 0) for p in current_prod_rev.index]
        })

        fig = px.bar(comparison_data, x='product', y=['current', 'previous'],
                     barmode='group',
                     height=200,
                     labels={'value': 'Revenue'},
                     color_discrete_sequence=['#2ecc71', '#95a5a6'])
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with tab4:
    st.header("AI-Powered Predictions")
    
    # Load models
    if 'models' not in st.session_state:
        st.session_state.models = load_models()
    
    if st.session_state.models is None:
        st.error("Failed to load prediction models. Please check the model files.")
        st.stop()
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # --- Market Growth Predictor ---
        st.subheader("🌍 Market Growth Predictor")
        with st.expander("Predict growth for next month", expanded=True):
            country_options = df['country'].unique()
            selected_country = st.selectbox("Select country", country_options, key='growth_country')
            
            if st.button("Predict Growth", key='growth_btn'):
                growth_model = st.session_state.models['growth_model']
                
                # Get the current month from the filtered data
                current_month = df_filtered['timestamp'].dt.month.max()
                
                # Prepare input data (exactly as in your training)
                input_data = pd.DataFrame({
                    'country': [selected_country],
                    'month': [current_month + 1]  # Next month
                })
                
                # One-hot encode exactly as done during training
                X = pd.get_dummies(input_data)
                
                # Ensure all expected columns are present
                expected_cols = growth_model.get_booster().feature_names
                for col in expected_cols:
                    if col not in X.columns:
                        X[col] = 0
                X = X[expected_cols]
                
                # Predict
                growth_prob = growth_model.predict_proba(X)[0][1]
                
                # Show result
                st.metric("Probability of Growth Next Month", f"{growth_prob:.1%}")
                
                # Historical context - using the filtered data
                country_data = df_filtered[df_filtered['country'] == selected_country]
                if not country_data.empty:
                    monthly_counts = country_data.groupby('month').size()
                    if len(monthly_counts) > 1:
                        last_month = monthly_counts.iloc[-1]
                        prev_month = monthly_counts.iloc[-2]
                        actual_growth = (last_month > prev_month)
                        st.metric("Actual Growth Last Month", 
                                "Yes" if actual_growth else "No",
                                delta=f"{(last_month - prev_month)/prev_month:.1%}" if prev_month > 0 else "N/A")
                        

                # Display model accuracy (from your training results)
                st.markdown("---")
                st.markdown("**Model Performance:**")
                st.metric("Training Accuracy", "90.9%")
                st.caption("Based on test set performance during model training")    

    with col2:
        # --- Page Engagement ---
        st.subheader("🔥 Page Engagement Predictor")
        with st.expander("Predict engagement for any page", expanded=True):
            page_options = df['page'].unique()
            selected_page = st.selectbox("Select a page to analyze", page_options, key='engagement_page')
            
            if st.button("Predict Engagement", key='engagement_btn'):
                engagement_model = st.session_state.models['engagement_model']
                scaler = st.session_state.models['engagement_scaler']
                le_page = st.session_state.models['page_encoder']
                
                # Prepare input
                page_encoded = le_page.transform([selected_page])
                X = np.array([[page_encoded[0]]])
                X_scaled = scaler.transform(X)
                
                # Predict
                pred_duration = engagement_model.predict(X_scaled)[0]
                
                # Show results in metrics
                st.metric("Predicted Average Session Duration", f"{pred_duration:.2f} seconds")
                
                # Compare with actual data
                actual_duration = df[df['page'] == selected_page]['session_duration'].mean()
                st.metric("Actual Average Duration", f"{actual_duration:.2f} seconds")
                
                diff = pred_duration - actual_duration
                st.metric("Difference", f"{diff:.2f} seconds", 
                         delta_color="inverse",
                         help="Positive values mean overprediction, negative means underprediction")
                
                # Display model accuracy
                st.markdown("---")
                st.markdown("**Model Performance:**")
                
                # Calculate R² score (example - replace with your actual metric)
                all_pages = df.groupby('page')['session_duration'].mean()
                baseline_error = np.mean((all_pages - np.mean(all_pages))**2)
                model_error = np.mean((pred_duration - actual_duration)**2)
                r2 = 1 - (model_error / baseline_error)
                
                st.metric("Prediction Accuracy", f"{max(0, r2)*100:.1f}%")
                st.caption("R² score comparing predictions to baseline")
