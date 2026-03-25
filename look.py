import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Cameroon NLP Dashboard", layout="wide")

st.title("Cameroon NLP Societal Analysis Since 2020")
st.markdown("Final Year Project - Clarckson Dobbit")
st.markdown("---")

# Load dataset
df = pd.read_csv("cameroon_societal_labeled_sentiment.csv")

# KPIs
total = len(df)
pos = len(df[df['sentiment'] == 'Positive'])
neg = len(df[df['sentiment'] == 'Negative'])
neu = len(df[df['sentiment'] == 'Neutral'])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Posts", total)
c2.metric("Positive %", round(pos/total*100,2))
c3.metric("Negative %", round(neg/total*100,2))
c4.metric("Neutral %", round(neu/total*100,2))

st.markdown("---")

# Sentiment chart
st.subheader("Sentiment Distribution")
fig1 = px.bar(df['sentiment'].value_counts().reset_index(),
              x='index', y='sentiment', color='index')
st.plotly_chart(fig1, use_container_width=True)

# Trend
if 'year' in df.columns:
    st.subheader("Sentiment Trend Over Time")
    trend = df.groupby(['year','sentiment']).size().reset_index(name='count')
    fig2 = px.line(trend, x='year', y='count', color='sentiment', markers=True)
    st.plotly_chart(fig2, use_container_width=True)

# Topics
if 'topic' in df.columns:
    st.subheader("Topics Analysis")
    fig3 = px.bar(df['topic'].value_counts().reset_index(),
                  x='index', y='topic')
    st.plotly_chart(fig3, use_container_width=True)