import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from course_mapper import CourseNameMapper  # Add this new import

# Set the dashboard title
st.title("📊 Learning Dashboard")

def calculate_business_days(start_date, end_date):
    """
    Calculate business days between two dates, handling invalid dates gracefully.
    """
    try:
        start_date = pd.to_datetime(start_date, errors='coerce')
        end_date = pd.to_datetime(end_date, errors='coerce')
        
        if pd.isna(start_date) or pd.isna(end_date):
            return 0
        
        business_days = pd.date_range(start=start_date, end=end_date, freq='B')
        return len(business_days)
    except Exception as e:
        print(f"Error calculating business days: {e}")
        return 0

# Note: We manually fixed French course names in the database files

def load_and_preprocess_data():
    """
    Load and preprocess all data files with improved course name handling
    """
    try:
        # Initialize course mapper
        from course_mapper import CourseNameMapper
        mapper = CourseNameMapper()
        
        # Load and standardize the courses data first
        df_courses = pd.read_excel('About the course.xlsx')
        df_courses = mapper.standardize_dataframe(df_courses)
        if df_courses.empty:
            raise Exception("Failed to load course master list")
        print("Successfully loaded courses data")
        
        # Load and standardize other dataframes
        df_enrollments = pd.read_csv('all_enrollments.csv', encoding='ISO-8859-1')
        df_enrollments = mapper.standardize_dataframe(df_enrollments)
        
        # Diagnostic prints
        print("\nENROLLMENT COUNT CHECK:")
        total_records = len(df_enrollments)
        unique_students = df_enrollments['Email'].nunique()
        print(f"Total enrollment records in file: {total_records}")
        print(f"Unique students: {unique_students}")
        
        print("\nCOURSE ENROLLMENT COUNTS:")
        for course, count in df_enrollments['Course Name'].value_counts().items():
            print(f"{course}: {count}")

        # Debug code for French courses
        print("\nDEBUG - French Course Data:")
        french_courses = [
            "Le paramétrage de DHIS2 agrégé",
            "Les fondamentaux de DHIS2 événements",
            "Les fondamentaux de la saisie et de la validation des données agrégées",
            "Les principes fondamentaux de DHIS2"
        ]
        
        # Check original data
        print("\nBefore standardization:")
        original_df = pd.read_csv('all_enrollments.csv', encoding='ISO-8859-1')
        for course in french_courses:
            variants = [col for col in original_df['Course Name'].unique() 
                       if any(word.lower() in col.lower() for word in ['dhis2', 'agrégé', 'événements'])]
            print(f"\nPossible variants found for {course}:")
            for variant in variants:
                count = len(original_df[original_df['Course Name'] == variant])
                print(f"'{variant}': {count} enrollments")

        # Check after standardization
        print("\nAfter standardization:")
        for course in french_courses:
            count = len(df_enrollments[df_enrollments['Course Name'] == course])
            print(f"'{course}': {count} enrollments")
        
        # After loading df_enrollments, add this diagnostic code
        print("\nCourse name variations found in enrollments:")
        for course in sorted(df_enrollments['Course Name'].unique()):
            if any(word in course.lower() for word in ['dhis2', 'événements', 'agrégé', 'fondamentaux']):
                count = len(df_enrollments[df_enrollments['Course Name'] == course])
                print(f"'{course}': {count} enrollments")
        
        df_registered = pd.read_csv('all_registered.csv', encoding='ISO-8859-1')
        df_registered = mapper.standardize_dataframe(df_registered)
        
        df_certificates = pd.read_csv('all_certificates.csv', encoding='ISO-8859-1')
        df_certificates = mapper.standardize_dataframe(df_certificates)
        
        df_support_tickets = pd.read_csv('jira_support_tickets.csv', encoding='ISO-8859-1')
        df_support_tickets = mapper.standardize_dataframe(df_support_tickets)
        
        # Convert date columns to datetime
        df_enrollments['Date Joined'] = pd.to_datetime(df_enrollments['Date Joined'], errors='coerce')
        df_registered['Date Joined'] = pd.to_datetime(df_registered['Date Joined'], errors='coerce')
        df_certificates['Created Date'] = pd.to_datetime(df_certificates['Created Date'], format='mixed', errors='coerce')
        df_support_tickets['Created'] = pd.to_datetime(df_support_tickets['Created'], errors='coerce')
        df_support_tickets['Updated'] = pd.to_datetime(df_support_tickets['Updated'], errors='coerce')
        
        # Clean any rows with invalid dates
        df_enrollments = df_enrollments.dropna(subset=['Date Joined'])
        df_registered = df_registered.dropna(subset=['Date Joined'])
        df_certificates = df_certificates.dropna(subset=['Created Date'])
        df_support_tickets = df_support_tickets.dropna(subset=['Created'])
        
        # Merge support tickets data with course data
        df_support_tickets = pd.merge(
            df_support_tickets,
            df_courses[['Course Name', 'Course ID']],
            how='left',
            on='Course Name'
        )
        
        return df_courses, df_enrollments, df_registered, df_certificates, df_support_tickets
        
    except Exception as e:
        print(f"Error in load_and_preprocess_data: {str(e)}")
        return None, None, None, None, None

# Load the data
df_courses, df_enrollments, df_registered, df_certificates, df_support_tickets = load_and_preprocess_data()

# Add date range filters
st.sidebar.markdown("### 📅 Date Filters")

# Get min and max dates from enrollment data
if df_enrollments is not None and not df_enrollments.empty:
    min_date = df_enrollments['Date Joined'].min()
    max_date = df_enrollments['Date Joined'].max()
else:
    min_date = pd.Timestamp.now()
    max_date = pd.Timestamp.now()

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

# Filter to include only the 15 core courses
core_courses = [
    "Introduction to DHIS2",
    "Aggregate Data Capture and Validation Fundamentals",
    "Aggregate Data Analysis Fundamentals",
    "Les principes fondamentaux de DHIS2",
    "Data Quality Level 2 Academy",
    "DHIS2 Events Fundamentals",
    "Aggregate Customization Fundamentals",
    "Planning and Budgeting DHIS2 Implementations",
    "Introduction à DHIS2",
    "Les fondamentaux de DHIS2 événements",
    "Introducción a DHIS2",
    "Le paramétrage de DHIS2 agrégé",
    "Fundamentos de Captura y Validación de Datos Agregados",
    "Fundamentos de Configuración de Datos Agregados",
    "Fundamentos de Análisis de Datos Agregados en DHIS2"
]

# Filter enrollments to core courses
df_enrollments = df_enrollments[df_enrollments['Course Name'].isin(core_courses)]
filtered_enrollments = filtered_enrollments[filtered_enrollments['Course Name'].isin(core_courses)]

# Calculate metrics
total_enrollments = len(df_enrollments)
unique_users = df_enrollments['Email'].nunique()
active_users_current = filtered_enrollments['Email'].nunique()
active_users_previous = df_enrollments[
    (df_enrollments['Date Joined'] >= start_of_previous_30_days) & 
    (df_enrollments['Date Joined'] <= end_of_previous_30_days)
]['Email'].nunique()
new_users_diff = active_users_current - active_users_previous

# Support tickets metrics
total_tickets = filtered_tickets.shape[0]
tickets_previous = df_support_tickets[
    (df_support_tickets['Created'] >= start_of_previous_30_days) & 
    (df_support_tickets['Created'] <= end_of_previous_30_days)
].shape[0]
tickets_diff = total_tickets - tickets_previous

# Update the metrics display
col1, col2, col3, col4 = st.columns(4)
col1.metric("👤 Total Enrollments", f"{total_enrollments:,}")
col2.metric("👥 Unique Users", f"{unique_users:,}")
col3.metric("📬 Tickets (Selected Period)", f"{total_tickets:,}")
col4.metric("📊 vs Previous Period", f"{tickets_diff:,}", f"{tickets_diff:+}")

# Support tickets metrics
total_tickets = filtered_tickets.shape[0]
tickets_previous = df_support_tickets[
    (df_support_tickets['Created'] >= start_of_previous_30_days) & 
    (df_support_tickets['Created'] <= end_of_previous_30_days)
].shape[0]
tickets_diff = total_tickets - tickets_previous

# Changed "Total Active Users" to "Total Enrollments"
col1.metric("👤 Total Enrollments", f"{total_enrollments:,}", f"New: {new_users_diff:+}")
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

# Updated Course Monitoring Section
st.markdown("### ⚠️ Course Monitoring")

# Calculate average metrics across all courses
avg_completion_rate = df_ranking['Completion Rate'].mean()

# Calculate average tickets per course
course_ticket_counts = filtered_tickets.groupby('Course ID').size()
avg_tickets = course_ticket_counts.mean()

# Calculate monitoring metrics for all courses
monitoring_data = []

for _, course in df_courses.iterrows():
    course_id = course['Course ID']
    
    # Get recent enrollments (30 days)
    recent_enrollments_30d = filtered_enrollments[
        filtered_enrollments['Course ID'] == course_id
    ]['Email'].nunique()
    
    # Get total tickets in last 30 days
    course_tickets = len(filtered_tickets[filtered_tickets['Course ID'] == course_id])
    
    # Calculate completion rate
    total_enrollments = df_enrollments[
        df_enrollments['Course ID'] == course_id
    ]['Email'].nunique()
    
    completions = df_certificates[
        df_certificates['Course ID'] == course_id
    ]['Email'].nunique()
    
    completion_rate = (completions / total_enrollments * 100) if total_enrollments > 0 else 0
    
    # Determine alert level based on new criteria
    alert_level = 'Low'
    if recent_enrollments_30d == 0 or completion_rate < avg_completion_rate or course_tickets > avg_tickets:
        alert_level = 'High'
    
    monitoring_data.append({
        'Course Name': course['Course Name'],
        'Alert Level': alert_level,
        'Enrollments (30 Days)': recent_enrollments_30d,
        'Tickets (30 Days)': course_tickets,
        'Completion Rate': completion_rate
    })

# Create monitoring DataFrame with reordered columns
df_monitoring = pd.DataFrame(monitoring_data)
df_monitoring = df_monitoring.sort_values('Alert Level', ascending=False)

# Format numeric columns
df_monitoring = df_monitoring.assign(
    **{
        'Enrollments (30 Days)': df_monitoring['Enrollments (30 Days)'].apply(lambda x: f'{int(x):,}'),
        'Tickets (30 Days)': df_monitoring['Tickets (30 Days)'].apply(lambda x: f'{int(x):,}'),
        'Completion Rate': df_monitoring['Completion Rate'].apply(lambda x: f'{x:.1f}%')
    }
)

# Define styling function for all columns
def style_dataframe(val, column_name):
    if column_name == 'Alert Level':
        if val == 'High':
            return 'background-color: #ffcdd2'
        return 'background-color: #c8e6c9'
    elif column_name == 'Enrollments (30 Days)':
        if val == '0':
            return 'background-color: #ffcdd2'
    elif column_name == 'Completion Rate':
        try:
            rate = float(val.strip('%'))
            if rate < avg_completion_rate:
                return 'background-color: #ffcdd2'
        except:
            pass
    elif column_name == 'Tickets (30 Days)':
        try:
            tickets = int(val.replace(',', ''))
            if tickets > avg_tickets:
                return 'background-color: #ffcdd2'
        except:
            pass
    return ''

# Apply the styling
styled_df = df_monitoring.style.apply(lambda x: [style_dataframe(val, col) for val, col in zip(x, df_monitoring.columns)], axis=1)
st.dataframe(styled_df)

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

# Calculate average resolution time
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

# Enrollment and Support Ticket Trends
st.markdown("#### 📈 Enrollment and Support Ticket Trends")

# Create two columns for charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
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
    # Daily support tickets trend
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

# Ticket Categories Distribution
st.markdown("#### 📊 Support Ticket Categories")
if len(course_tickets_period) > 0:
    category_dist = course_tickets_period['Category'].value_counts()
    fig_category = px.pie(
        values=category_dist.values,
        names=category_dist.index,
        title='Support Ticket Categories Distribution'
    )
    fig_category.update_layout(
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=30, l=20, r=20, b=20)
    )
    fig_category.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(colors=['#2E86C1', '#E74C3C', '#2ECC71', '#F1C40F', '#9B59B6'])
    )
    st.plotly_chart(fig_category, use_container_width=True)
else:
    st.info("No support tickets data available for the selected period.")

# Calculate resolution times if there are resolved tickets
if len(course_tickets_period) > 0:
    resolved_tickets = course_tickets_period[course_tickets_period['Status'] == 'Done']
    if len(resolved_tickets) > 0:
        resolution_times = []
        for _, ticket in resolved_tickets.iterrows():
            days = calculate_business_days(ticket['Created'], ticket['Updated'])
            resolution_times.append(days)
        
        avg_resolution = np.mean(resolution_times)
        st.metric("Average Resolution Time (Business Days)", f"{avg_resolution:.1f} days")