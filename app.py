import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Set the dashboard title
st.title("📊 Learning Dashboard")

def calculate_business_days(start_date, end_date):
    """
    Calculate business days between two dates, handling invalid dates gracefully.
    """
    try:
        # Convert to datetime if not already
        start_date = pd.to_datetime(start_date, errors='coerce')
        end_date = pd.to_datetime(end_date, errors='coerce')
        
        # Check if either date is NaT
        if pd.isna(start_date) or pd.isna(end_date):
            return 0
        
        # Calculate business days
        business_days = pd.date_range(start=start_date, end=end_date, freq='B')
        return len(business_days)
    except Exception as e:
        print(f"Error calculating business days: {e}")
        return 0

# Load and preprocess the data
def load_and_preprocess_data():
    # Load the data
    df_courses = pd.read_excel('About the course.xlsx')
    df_enrollments = pd.read_csv('all_enrollments.csv')
    df_certificates = pd.read_csv('all_certificates.csv')
    df_support_tickets = pd.read_csv('jira_support_tickets.csv', encoding='ISO-8859-1')
    
    # Convert date columns to datetime with more flexible parsing
    df_enrollments['Date Joined'] = pd.to_datetime(df_enrollments['Date Joined'], errors='coerce')
    df_certificates['Created Date'] = pd.to_datetime(df_certificates['Created Date'], format='mixed', errors='coerce')
    df_support_tickets['Created'] = pd.to_datetime(df_support_tickets['Created'], errors='coerce')
    df_support_tickets['Updated'] = pd.to_datetime(df_support_tickets['Updated'], errors='coerce')
    
    # Clean any rows with invalid dates
    df_enrollments = df_enrollments.dropna(subset=['Date Joined'])
    df_certificates = df_certificates.dropna(subset=['Created Date'])
    df_support_tickets = df_support_tickets.dropna(subset=['Created'])
    
    # Merge support tickets data with course data
    df_support_tickets = pd.merge(
        df_support_tickets,
        df_courses[['Course Name', 'Course ID']],
        how='left',
        left_on='Course Name',
        right_on='Course Name'
)   
      
    return df_courses, df_enrollments, df_certificates, df_support_tickets

# Load the data
df_courses, df_enrollments, df_certificates, df_support_tickets = load_and_preprocess_data()

# Add date range filters
st.sidebar.markdown("### 📅 Date Filters")

# Get min and max dates from enrollment data
min_date = df_enrollments['Date Joined'].min()
max_date = df_enrollments['Date Joined'].max()

# Add date range selector with default values
start_date = st.sidebar.date_input(
    "Start Date",
    value=max_date.date() - pd.Timedelta(days=30),
    min_value=min_date.date(),
    max_value=max_date.date()
)

end_date = st.sidebar.date_input(
    "End Date",
    value=max_date.date(),
    min_value=min_date.date(),
    max_value=max_date.date()
)

# Filter dataframes based on date selection
filtered_enrollments = df_enrollments[
    (df_enrollments['Date Joined'] >= pd.Timestamp(start_date)) & 
    (df_enrollments['Date Joined'] <= pd.Timestamp(end_date))
]

filtered_certificates = df_certificates[
    (df_certificates['Created Date'] >= pd.Timestamp(start_date)) & 
    (df_certificates['Created Date'] <= pd.Timestamp(end_date))
]

filtered_tickets = df_support_tickets[
    (df_support_tickets['Created'] >= pd.Timestamp(start_date)) & 
    (df_support_tickets['Created'] <= pd.Timestamp(end_date))
]
# Display the last update date
if pd.notnull(max_date):
    last_update_str = max_date.strftime('%B %d, %Y')
else:
    last_update_str = "No valid date available"
st.markdown(f"**📅 Data Last Updated:** {last_update_str}")

# Calculate date ranges for comparisons
last_update = pd.Timestamp(end_date)
start_of_last_30_days = pd.Timestamp(end_date) - pd.Timedelta(days=30)
start_of_previous_30_days = pd.Timestamp(start_of_last_30_days) - pd.Timedelta(days=30)
end_of_previous_30_days = pd.Timestamp(start_of_last_30_days) - pd.Timedelta(days=1)

# Aggregator Section
st.markdown("### 🔍 Aggregated Insights")
col1, col2, col3, col4 = st.columns(4)

# Calculate metrics
active_users_current = filtered_enrollments['Email'].nunique()
active_users_previous = df_enrollments[
    (df_enrollments['Date Joined'] >= start_of_previous_30_days) & 
    (df_enrollments['Date Joined'] <= end_of_previous_30_days)
]['Email'].nunique()
new_users_diff = active_users_current - active_users_previous
total_active_users = df_enrollments['Email'].nunique()

# Support tickets metrics
total_tickets = filtered_tickets.shape[0]
tickets_previous = df_support_tickets[
    (df_support_tickets['Created'] >= start_of_previous_30_days) & 
    (df_support_tickets['Created'] <= end_of_previous_30_days)
].shape[0]
tickets_diff = total_tickets - tickets_previous

# Display metrics
col1.metric("👤 Total Active Users", f"{total_active_users:,}", f"New: {new_users_diff:+}")
col2.metric("📮 Total Support Tickets", f"{df_support_tickets.shape[0]:,}")
col3.metric("📬 Tickets (Selected Period)", f"{total_tickets:,}")
col4.metric("📊 vs Previous Period", f"{tickets_diff:,}", f"{tickets_diff:+}")

# Course Completion Rate Ranking
st.markdown("### 🏆 Course Completion Rate Ranking")

# Calculate completion rates for all courses
completion_ranking = []
for _, course in df_courses.iterrows():
    total_enrollments = df_enrollments[
        df_enrollments['Course ID'] == course['Course ID']
    ]['Email'].nunique()
    
    total_completions = df_certificates[
        df_certificates['Course ID'] == course['Course ID']
    ]['Email'].nunique()
    
    recent_enrollments = filtered_enrollments[
        filtered_enrollments['Course ID'] == course['Course ID']
    ]['Email'].nunique()
    
    completion_rate = (total_completions / total_enrollments * 100) if total_enrollments > 0 else 0
    
    completion_ranking.append({
        'Course Name': course['Course Name'],
        'Total Enrollments': total_enrollments,
        'Total Completions': total_completions,
        'Completion Rate': completion_rate,
        'Recent Enrollments (Last 30 Days)': recent_enrollments
    })

df_ranking = pd.DataFrame(completion_ranking)
df_ranking = df_ranking.sort_values('Completion Rate', ascending=False)
df_ranking['Rank'] = range(1, len(df_ranking) + 1)

# Create a styled table with completion rates
fig_ranking = go.Figure(data=[go.Table(
    header=dict(
        values=['Rank', 'Course Name', 'Completion Rate', 'Total Enrollments', 'Recent Enrollments (Last 30 Days)'],
        fill_color='#2E86C1',
        font=dict(color='white'),
        align='left'
    ),
    cells=dict(
        values=[
            df_ranking['Rank'],
            df_ranking['Course Name'],
            df_ranking['Completion Rate'].apply(lambda x: f'{x:.1f}%'),
            df_ranking['Total Enrollments'].apply(lambda x: f'{x:,}'),
            df_ranking['Recent Enrollments (Last 30 Days)'].apply(lambda x: f'{x:,}')
        ],
        fill_color=[
            ['white']*len(df_ranking),
            ['white']*len(df_ranking),
            [f'rgba(46, 134, 193, {rate/100})' for rate in df_ranking['Completion Rate']],
            ['white']*len(df_ranking),
            ['white']*len(df_ranking)
        ],
        align='left'
    )
)])

fig_ranking.update_layout(
    height=400,
    margin=dict(l=0, r=0, t=0, b=0)
)

st.plotly_chart(fig_ranking, use_container_width=True)

# Course Monitoring Section
st.markdown("### ⚠️ Course Monitoring")

# Calculate monitoring metrics for all courses
monitoring_data = []
seven_days_ago = pd.Timestamp.now() - pd.Timedelta(days=7)

for _, course in df_courses.iterrows():
    course_id = course['Course ID']
    
    # Get recent enrollments (7 days)
    recent_enrollments_7d = df_enrollments[
        (df_enrollments['Course ID'] == course_id) &
        (df_enrollments['Date Joined'] >= seven_days_ago)
    ]['Email'].nunique()
    
    # Get recent enrollments (30 days)
    recent_enrollments_30d = filtered_enrollments[
        filtered_enrollments['Course ID'] == course_id
    ]['Email'].nunique()
    
    # Get tickets
    course_tickets = filtered_tickets[
        filtered_tickets['Course ID'] == course_id
    ]
    open_tickets = len(course_tickets[course_tickets['Status'] == 'In Progress'])
    total_tickets = len(course_tickets)
    
    # Calculate completion rate
    total_enrollments = df_enrollments[
        df_enrollments['Course ID'] == course_id
    ]['Email'].nunique()
    
    completions = df_certificates[
        df_certificates['Course ID'] == course_id
    ]['Email'].nunique()
    
    completion_rate = (completions / total_enrollments * 100) if total_enrollments > 0 else 0
    
    # Determine alert level and reasons
    alert_reasons = []
    if recent_enrollments_7d == 0:
        alert_reasons.append("No enrollments in last 7 days")
    if open_tickets > 5:
        alert_reasons.append("High number of open tickets")
    if completion_rate < 30:
        alert_reasons.append("Low completion rate")
    
    alert_level = 'High' if len(alert_reasons) >= 2 else 'Medium' if len(alert_reasons) == 1 else 'Low'
    
    monitoring_data.append({
        'Course Name': course['Course Name'],
        'Enrollments (7 Days)': recent_enrollments_7d,
        'Enrollments (30 Days)': recent_enrollments_30d,
        'Open Tickets': open_tickets,
        'Total Tickets': total_tickets,
        'Completion Rate': completion_rate,
        'Alert Level': alert_level,
        'Alert Reasons': '; '.join(alert_reasons) if alert_reasons else 'No issues'
    })

# Create monitoring DataFrame
df_monitoring = pd.DataFrame(monitoring_data)
df_monitoring = df_monitoring.sort_values(['Alert Level', 'Open Tickets', 'Completion Rate'], 
                                        ascending=[False, False, True])

# Format numeric columns before displaying
df_monitoring = df_monitoring.assign(
    **{
        'Enrollments (7 Days)': df_monitoring['Enrollments (7 Days)'].apply(lambda x: f'{int(x):,}'),
        'Enrollments (30 Days)': df_monitoring['Enrollments (30 Days)'].apply(lambda x: f'{int(x):,}'),
        'Open Tickets': df_monitoring['Open Tickets'].apply(lambda x: f'{int(x):,}'),
        'Total Tickets': df_monitoring['Total Tickets'].apply(lambda x: f'{int(x):,}'),
        'Completion Rate': df_monitoring['Completion Rate'].apply(lambda x: f'{x:.1f}%')
    }
)

# Use a simpler styling approach
def highlight_alert_level(val):
    if val == 'High':
        return 'background-color: #ffcdd2'
    elif val == 'Medium':
        return 'background-color: #fff176'
    elif val == 'Low':
        return 'background-color: #c8e6c9'
    return ''

# Apply the styling
st.dataframe(
    df_monitoring.style.apply(
        lambda x: ['background-color: transparent' if i != 'Alert Level' else highlight_alert_level(x['Alert Level']) 
                  for i in df_monitoring.columns],
        axis=1
    )
)

# Support Ticket Analysis Section
st.markdown("### 🎫 Support Ticket Analysis")

# Add ticket category filter
categories = ['All Categories'] + list(filtered_tickets['Category'].unique())
selected_category = st.selectbox('Filter by Ticket Category', categories)

# Filter tickets by category
if selected_category != 'All Categories':
    analysis_tickets = filtered_tickets[filtered_tickets['Category'] == selected_category]
else:
    analysis_tickets = filtered_tickets

# Create metrics for ticket overview
ticket_col1, ticket_col2, ticket_col3, ticket_col4 = st.columns(4)

# Calculate ticket metrics
total_period_tickets = len(analysis_tickets)
in_progress_tickets = len(analysis_tickets[analysis_tickets['Status'] == 'In Progress'])
resolved_tickets = len(analysis_tickets[analysis_tickets['Status'] == 'Done'])

# Calculate average resolution time for resolved tickets
resolved_ticket_times = []
for _, ticket in analysis_tickets[analysis_tickets['Status'] == 'Done'].iterrows():
    resolution_days = calculate_business_days(ticket['Created'], ticket['Updated'])
    resolved_ticket_times.append(resolution_days)

avg_resolution_time = np.mean(resolved_ticket_times) if resolved_ticket_times else 0

# Display ticket metrics
ticket_col1.metric("Total Tickets", f"{total_period_tickets:,}")
ticket_col2.metric("In Progress", f"{in_progress_tickets:,}")
ticket_col3.metric("Resolved", f"{resolved_tickets:,}")
ticket_col4.metric("Avg. Resolution Time", f"{avg_resolution_time:.1f} days")

# Create two columns for charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Status Distribution
    status_dist = analysis_tickets['Status'].value_counts()
    fig_status = px.pie(
        values=status_dist.values,
        names=status_dist.index,
        title='Ticket Status Distribution'
    )
    fig_status.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_status, use_container_width=True)

with chart_col2:
    # Category Distribution (if showing all categories)
    if selected_category == 'All Categories':
        category_dist = analysis_tickets['Category'].value_counts()
        fig_category = px.pie(
            values=category_dist.values,
            names=category_dist.index,
            title='Ticket Category Distribution'
        )
        fig_category.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_category, use_container_width=True)

# Course Support Tickets Analysis
course_tickets = analysis_tickets.groupby('Course Name').agg({
    'Status': 'count',
    'Created': 'count'
}).reset_index()
course_tickets.columns = ['Course Name', 'Total Tickets', 'Created']
course_tickets = course_tickets.sort_values('Total Tickets', ascending=False)

# Show course ticket distribution
fig_course_tickets = px.bar(
    course_tickets,
    x='Total Tickets',
    y='Course Name',
    title='Support Tickets by Course',
    orientation='h',
    color='Total Tickets',
    color_continuous_scale='Reds'
)
fig_course_tickets.update_layout(
    height=400,
    yaxis={'categoryorder': 'total ascending'},
    showlegend=False
)
st.plotly_chart(fig_course_tickets, use_container_width=True)

# Ticket Trend Analysis
st.markdown("#### 📈 Support Tickets Trend")
daily_tickets = analysis_tickets.groupby(
    analysis_tickets['Created'].dt.date
).size().reset_index(name='Tickets')

fig_trend = px.line(
    daily_tickets,
    x='Created',
    y='Tickets',
    title='Daily Support Tickets',
    labels={'Created': 'Date', 'Tickets': 'Number of Tickets'}
)
fig_trend.update_traces(line_color='#E74C3C', line_width=2, line_shape='spline')
fig_trend.update_layout(
    height=300,
    showlegend=False,
    hovermode='x unified',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)
st.plotly_chart(fig_trend, use_container_width=True)

# Course-specific analysis
st.markdown("### 📋 Course-Specific Analysis")
selected_course = st.selectbox('Select a Course', df_courses['Course Name'].unique())
course_data = df_courses[df_courses['Course Name'] == selected_course].iloc[0]

# Create three columns for metadata
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"**⏳ Course Duration**\n{course_data['Course duration']} weeks")
with col2:
    st.info(f"**🧩 Number of Modules**\n{course_data['Number of modules']}")
with col3:
    st.info(f"**📊 Passing Score**\n{course_data['Passing score']}")

# Calculate metrics for selected course
course_enrollments_all = df_enrollments[
    df_enrollments['Course ID'] == course_data['Course ID']
]
course_certificates_all = df_certificates[
    df_certificates['Course ID'] == course_data['Course ID']
]
course_tickets_all = df_support_tickets[
    df_support_tickets['Course ID'] == course_data['Course ID']
]

# Calculate period-specific metrics
course_enrollments_period = filtered_enrollments[
    filtered_enrollments['Course ID'] == course_data['Course ID']
]
course_certificates_period = filtered_certificates[
    filtered_certificates['Course ID'] == course_data['Course ID']
]
course_tickets_period = filtered_tickets[
    filtered_tickets['Course ID'] == course_data['Course ID']
]

# Calculate different metrics
metrics = {
    'All Time': {
        'Enrollments': course_enrollments_all['Email'].nunique(),
        'Completions': course_certificates_all['Email'].nunique(),
        'Support Tickets': len(course_tickets_all),
        'Open Tickets': len(course_tickets_all[course_tickets_all['Status'] == 'In Progress'])
    },
    'Selected Period': {
        'Enrollments': course_enrollments_period['Email'].nunique(),
        'Completions': course_certificates_period['Email'].nunique(),
        'Support Tickets': len(course_tickets_period),
        'Open Tickets': len(course_tickets_period[course_tickets_period['Status'] == 'In Progress'])
    }
}

# Calculate completion rates
metrics['All Time']['Completion Rate'] = (
    (metrics['All Time']['Completions'] / metrics['All Time']['Enrollments'] * 100)
    if metrics['All Time']['Enrollments'] > 0 else 0
)
metrics['Selected Period']['Completion Rate'] = (
    (metrics['Selected Period']['Completions'] / metrics['Selected Period']['Enrollments'] * 100)
    if metrics['Selected Period']['Enrollments'] > 0 else 0
)

# Display metrics in expandable sections
st.markdown("#### Course Performance Metrics")

# All-time metrics
with st.expander("📈 Overall Course Statistics", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Enrollments", f"{metrics['All Time']['Enrollments']:,}")
    col2.metric("Total Completions", f"{metrics['All Time']['Completions']:,}")
    col3.metric("Completion Rate", f"{metrics['All Time']['Completion Rate']:.1f}%")
    col4.metric("Total Support Tickets", f"{metrics['All Time']['Support Tickets']:,}")

# Selected period metrics
with st.expander("🎯 Selected Period Performance", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Period Enrollments", f"{metrics['Selected Period']['Enrollments']:,}")
    col2.metric("Period Completions", f"{metrics['Selected Period']['Completions']:,}")
    col3.metric("Period Completion Rate", f"{metrics['Selected Period']['Completion Rate']:.1f}%")
    col4.metric("Period Support Tickets", f"{metrics['Selected Period']['Support Tickets']:,}")

# Enrollment Trend Analysis
st.markdown("#### 📈 Enrollment and Support Ticket Trends")

# Create two columns for charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Daily enrollments trend
    daily_enrollments = course_enrollments_period.groupby(
        course_enrollments_period['Date Joined'].dt.date
    ).size().reset_index(name='Enrollments')

    if not daily_enrollments.empty:
        fig_enrollments = px.line(
            daily_enrollments,
            x='Date Joined',
            y='Enrollments',
            title='Daily Enrollments',
            labels={'Date Joined': 'Date', 'Enrollments': 'Number of Enrollments'}
        )
        fig_enrollments.update_traces(
            line_color='#2E86C1',
            line_width=2,
            line_shape='spline'
        )
        fig_enrollments.update_layout(
            height=300,
            showlegend=False,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_enrollments, use_container_width=True)
    else:
        st.info("No enrollment data available for the selected period.")

with chart_col2:
    # Support tickets trend
    daily_tickets = course_tickets_period.groupby(
        course_tickets_period['Created'].dt.date
    ).size().reset_index(name='Tickets')

    if not daily_tickets.empty:
        fig_tickets = px.line(
            daily_tickets,
            x='Created',
            y='Tickets',
            title='Daily Support Tickets',
            labels={'Created': 'Date', 'Tickets': 'Number of Tickets'}
        )
        fig_tickets.update_traces(
            line_color='#E74C3C',
            line_width=2,
            line_shape='spline'
        )
        fig_tickets.update_layout(
            height=300,
            showlegend=False,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_tickets, use_container_width=True)
    else:
        st.info("No support tickets data available for the selected period.")

# Support Ticket Details for Selected Course
if len(course_tickets_period) > 0:
    st.markdown("#### 🎫 Support Ticket Details")
    
    # Show ticket status distribution
    ticket_status = course_tickets_period['Status'].value_counts()
    fig_status = px.pie(
        values=ticket_status.values,
        names=ticket_status.index,
        title='Ticket Status Distribution'
    )
    fig_status.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_status, use_container_width=True)

    # Calculate and display resolution times
    resolved_tickets = course_tickets_period[course_tickets_period['Status'] == 'Done']
    if len(resolved_tickets) > 0:
        resolution_times = []
        for _, ticket in resolved_tickets.iterrows():
            days = calculate_business_days(ticket['Created'], ticket['Updated'])
            resolution_times.append(days)
        
        avg_resolution = np.mean(resolution_times)
        st.metric("Average Resolution Time (Business Days)", f"{avg_resolution:.1f} days")