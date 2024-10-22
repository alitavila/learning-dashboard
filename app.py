import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set the dashboard title
st.title("📊 Learning Dashboard")

# Load the course data, enrollment/certificates, and support tickets
df_courses = pd.read_excel('About the course.xlsx')
df_enrollments = pd.read_csv('all_enrollments.csv')
df_certificates = pd.read_csv('all_certificates.csv')

# Convert 'Created Date' column in certificates to datetime
df_certificates['Created Date'] = pd.to_datetime(df_certificates['Created Date'], errors='coerce')

df_support_tickets = pd.read_csv('jira_support_tickets.csv', encoding='ISO-8859-1')

# Merge support tickets data with the course data to add 'Course ID'
df_support_tickets = pd.merge(df_support_tickets, df_courses[['Course Name', 'Course ID']], how='left', left_on='Course Name', right_on='Course Name')

# Convert 'Date Joined' and 'Created' columns to datetime
df_enrollments['Date Joined'] = pd.to_datetime(df_enrollments['Date Joined'], errors='coerce')
df_support_tickets['Created'] = pd.to_datetime(df_support_tickets['Created'], errors='coerce')

# Get the latest available date from the enrollment data
last_update = df_enrollments['Date Joined'].max()

# Ensure it's not a NaT (Not a Time) and convert to string format
if pd.notnull(last_update):
    last_update_str = last_update.strftime('%B %d, %Y')
else:
    last_update_str = "No valid date available"

# Display the last update date on the dashboard
st.markdown(f"**📅 Data Last Updated:** {last_update_str}")

# Calculate the start and end dates for the last 30 days and the previous 30 days
start_of_last_30_days = last_update - pd.DateOffset(days=30)
start_of_previous_30_days = last_update - pd.DateOffset(days=60)
end_of_previous_30_days = last_update - pd.DateOffset(days=31)

# Aggregator Section with column layout
st.markdown("### 🔍 Aggregated Insights")

# Create a 4-column layout for the aggregated insights
col1, col2, col3, col4 = st.columns(4)

# Total number of active users (new users in the last 30 days)
active_users_last_30 = df_enrollments[df_enrollments['Date Joined'] >= start_of_last_30_days]['Email'].nunique()
active_users_previous_30 = df_enrollments[(df_enrollments['Date Joined'] >= start_of_previous_30_days) & (df_enrollments['Date Joined'] <= end_of_previous_30_days)]['Email'].nunique()
new_users_last_30 = active_users_last_30 - active_users_previous_30
active_users = df_enrollments['Email'].nunique()  # Total active users
col1.metric("👤 Total Active Users", active_users, f"New: {new_users_last_30}")

# Support tickets metrics
total_tickets = df_support_tickets.shape[0]
tickets_last_30 = df_support_tickets[df_support_tickets['Created'] >= start_of_last_30_days].shape[0]
tickets_previous_30 = df_support_tickets[(df_support_tickets['Created'] >= start_of_previous_30_days) & 
                                       (df_support_tickets['Created'] <= end_of_previous_30_days)].shape[0]
tickets_diff = tickets_last_30 - tickets_previous_30
col2.metric("📮 Total Support Tickets", total_tickets)
col3.metric("📬 Tickets (Last 30 Days)", tickets_last_30)
col4.metric("📊 vs Previous 30 Days", tickets_diff, f"{tickets_diff:+}")

# Calculate completion rates for course ranking
course_completion_rates = []
for _, course in df_courses.iterrows():
    course_enrollments = df_enrollments[df_enrollments['Course ID'] == course['Course ID']]['Email'].nunique()
    course_certificates = df_certificates[df_certificates['Course ID'] == course['Course ID']]['Email'].nunique()
    completion_rate = (course_certificates / course_enrollments * 100) if course_enrollments > 0 else 0
    course_completion_rates.append({
        'Course ID': course['Course ID'],
        'Course Name': course['Course Name'],
        'Completion Rate': completion_rate
    })

df_completion_rates = pd.DataFrame(course_completion_rates)
df_completion_rates = df_completion_rates.sort_values('Completion Rate', ascending=False)
df_completion_rates['Ranking'] = range(1, len(df_completion_rates) + 1)

# Add course enrollments visualization using sunburst chart
st.markdown("### 📊 Course Enrollments in Last 30 Days")

course_enrollments_last_30 = []
for _, course in df_courses.iterrows():
    enrollments = df_enrollments[(df_enrollments['Course ID'] == course['Course ID']) & 
                                (df_enrollments['Date Joined'] >= start_of_last_30_days)].shape[0]
    if enrollments > 0:  # Only include courses with enrollments
        course_enrollments_last_30.append({
            'Course Name': course['Course Name'],
            'Enrollments': enrollments,
            'Percentage': 0  # Will be calculated next
        })

# Calculate percentages
total_enrollments = sum(item['Enrollments'] for item in course_enrollments_last_30)
for item in course_enrollments_last_30:
    item['Percentage'] = (item['Enrollments'] / total_enrollments * 100) if total_enrollments > 0 else 0

# Create DataFrame and sort by enrollments
df_course_enrollments = pd.DataFrame(course_enrollments_last_30)
df_course_enrollments = df_course_enrollments.sort_values('Enrollments', ascending=False)

# Create a bar chart instead of treemap
fig_enrollments = px.bar(df_course_enrollments,
                        x='Course Name',
                        y='Enrollments',
                        title='Enrollments by Course (Last 30 Days)',
                        color='Enrollments',
                        color_continuous_scale='Blues')

# Customize the layout
fig_enrollments.update_layout(
    xaxis_tickangle=-45,
    xaxis_title="Course Name",
    yaxis_title="Number of Enrollments",
    height=500
)

# Display the chart
st.plotly_chart(fig_enrollments, use_container_width=True)

# Also display the data in a table format below the chart
st.markdown("### 📊 Detailed Enrollment Numbers")
st.dataframe(df_course_enrollments[['Course Name', 'Enrollments', 'Percentage']].style.format({
    'Enrollments': '{:,.0f}',
    'Percentage': '{:.1f}%'
}))

# Section for Courses with a Higher Enrollment Trend
st.markdown("### 📈 Courses with Higher Enrollment Trend in the Last 30 Days")
course_trend_data = []

for _, course in df_courses.iterrows():
    course_id = course['Course ID']
    course_name = course['Course Name']
    
    enrollments_last_30 = df_enrollments[(df_enrollments['Course ID'] == course_id) & 
                                        (df_enrollments['Date Joined'] >= start_of_last_30_days)].shape[0]
    enrollments_previous_30 = df_enrollments[(df_enrollments['Course ID'] == course_id) & 
                                           (df_enrollments['Date Joined'] >= start_of_previous_30_days) & 
                                           (df_enrollments['Date Joined'] <= end_of_previous_30_days)].shape[0]
    
    if enrollments_last_30 > enrollments_previous_30:
        increase_percentage = ((enrollments_last_30 - enrollments_previous_30) / enrollments_previous_30) * 100 if enrollments_previous_30 > 0 else float('inf')
        increase_absolute = enrollments_last_30 - enrollments_previous_30
        course_trend_data.append({
            'Course Name': course_name,
            'Enrollments Last 30 Days': enrollments_last_30,
            'Enrollments Previous 30 Days': enrollments_previous_30,
            'Absolute Increase': increase_absolute,
            'Percentage Increase': increase_percentage
        })

# Sort courses by enrollment trend in descending order
if course_trend_data:
    trend_df = pd.DataFrame(course_trend_data).sort_values(by='Absolute Increase', ascending=False)
    st.dataframe(trend_df)
else:
    st.write("No courses show a higher enrollment trend in the last 30 days compared to the previous 30 days.")

# Dropdown to select a course
selected_course = st.selectbox('Select a Course', df_courses['Course Name'].unique())

# Filter course metadata
course_data = df_courses[df_courses['Course Name'] == selected_course].iloc[0]

# Display course metadata with better formatting
st.markdown(f"### 📋 Metrics for {course_data['Course Name']}")
st.write(f"**⏳ Course Duration**: {course_data['Course duration']} weeks")
st.write(f"**🧩 Number of Modules**: {course_data['Number of modules']}")
st.write(f"**📅 Start Date**: {course_data['Start date']}")
st.write(f"**📊 Passing Score**: {course_data['Passing score']}")

# Filter enrollments, certificates, and support tickets for the selected course
filtered_enrollments = df_enrollments[df_enrollments['Course ID'] == course_data['Course ID']]
filtered_certificates = df_certificates[df_certificates['Course ID'] == course_data['Course ID']]
filtered_support_tickets = df_support_tickets[df_support_tickets['Course ID'] == course_data['Course ID']]

# Calculate metrics for the selected course
enrolled_students_last_30 = filtered_enrollments[filtered_enrollments['Date Joined'] >= start_of_last_30_days]['Email'].nunique()
enrolled_students_previous_30 = filtered_enrollments[(filtered_enrollments['Date Joined'] >= start_of_previous_30_days) & 
                                                   (filtered_enrollments['Date Joined'] <= end_of_previous_30_days)]['Email'].nunique()

total_enrolled_students = filtered_enrollments['Email'].nunique()
total_enrollments = df_enrollments['Email'].nunique()
enrollment_share = (total_enrolled_students / total_enrollments * 100) if total_enrollments > 0 else 0
enrollment_share_previous = (enrolled_students_previous_30 / total_enrollments * 100) if total_enrollments > 0 else 0

# Calculate completion rates for the current period
completion_rate = (filtered_certificates['Email'].nunique() / total_enrolled_students * 100) if total_enrolled_students > 0 else 0
completion_rate_previous = (filtered_certificates[(filtered_certificates['Created Date'] >= start_of_previous_30_days) & 
                                                (filtered_certificates['Created Date'] <= end_of_previous_30_days)]['Email'].nunique() / 
                           enrolled_students_previous_30 * 100) if enrolled_students_previous_30 > 0 else 0

# Get course ranking from the pre-calculated rankings
course_ranking = df_completion_rates[df_completion_rates['Course ID'] == course_data['Course ID']].iloc[0]['Ranking']

# Calculate support ticket metrics
support_tickets_total = filtered_support_tickets.shape[0]
support_tickets_last_30 = filtered_support_tickets[filtered_support_tickets['Created'] >= start_of_last_30_days].shape[0]
support_tickets_previous_30 = filtered_support_tickets[(filtered_support_tickets['Created'] >= start_of_previous_30_days) & 
                                                     (filtered_support_tickets['Created'] <= end_of_previous_30_days)].shape[0]

# Create disaggregated data DataFrame with separate columns for current and previous periods
disaggregated_data = pd.DataFrame({
    'Metric': [
        'Enrolled Students (Last 30 Days)',
        'Total Enrollment Share',
        'Course Completion Rate',
        'Completion Rate Ranking',
        'Support Tickets (Total)',
        'Support Tickets (Last 30 Days)'
    ],
    'Current Period': [
        f"{enrolled_students_last_30:,}",
        f"{enrollment_share:.2f}%",
        f"{completion_rate:.2f}%",
        f"#{course_ranking} of {len(df_courses)}",
        f"{support_tickets_total:,}",
        f"{support_tickets_last_30:,}"
    ],
    'Previous Period': [
        f"{enrolled_students_previous_30:,}",
        f"{enrollment_share_previous:.2f}%",
        f"{completion_rate_previous:.2f}%",
        "N/A",
        "N/A",
        f"{support_tickets_previous_30:,}"
    ]
})

# Display the disaggregated data as a table
st.markdown("### 📊 Selected Course Data")
st.table(disaggregated_data)

# Create enrollment trend visualization
st.markdown("### 📈 Enrollment Trend (Last 30 Days)")
filtered_enrollments_30_days = filtered_enrollments[(filtered_enrollments['Date Joined'] >= start_of_last_30_days) & 
                                                  (filtered_enrollments['Date Joined'] <= last_update)]
enrollments_over_time = filtered_enrollments_30_days.groupby(filtered_enrollments_30_days['Date Joined'].dt.date).size().reset_index(name='Enrollments')
fig_enrollments = px.line(enrollments_over_time, x='Date Joined', y='Enrollments', 
                         title='Daily Enrollments',
                         color_discrete_sequence=["#636EFA"])
st.plotly_chart(fig_enrollments)

# Create support tickets trend visualization
st.markdown("### 📮 Support Tickets Trend (Last 60 Days)")
filtered_tickets_60_days = filtered_support_tickets[(filtered_support_tickets['Created'] >= start_of_previous_30_days) & 
                                                  (filtered_support_tickets['Created'] <= last_update)]
tickets_over_time = filtered_tickets_60_days.groupby(filtered_tickets_60_days['Created'].dt.date).size().reset_index(name='Tickets')
fig_tickets = px.line(tickets_over_time, x='Created', y='Tickets', 
                     title='Daily Support Tickets',
                     color_discrete_sequence=["#EF553B"])
st.plotly_chart(fig_tickets)