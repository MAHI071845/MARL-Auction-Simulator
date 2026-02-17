import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

st.set_page_config(page_title="MARL Auction Simulator", layout="wide")

st.title("🏏 Multi-Agent RL Auction Simulator")
st.markdown("## 🎯 Live Auction with Trained RL Agent")

# Load players
players = pd.read_csv("data/players.csv")

# Load trained model
@st.cache_resource
def load_model():
    return PPO.load("marl_agent")

model = load_model()

# Session state
if "budget" not in st.session_state:
    st.session_state.budget = 100
    st.session_state.team = []
    st.session_state.idx = 0

st.subheader("📋 Player Pool")
st.dataframe(players)

st.subheader("💰 Remaining Budget")
st.write(st.session_state.budget)

# Auction button
if st.button("Next RL Bid"):

    if st.session_state.idx < len(players):

        player = players.iloc[st.session_state.idx]

        obs = np.array([
            player.rating,
            player.base_price,
            st.session_state.budget
        ], dtype=np.float32)

        action, _ = model.predict(obs)

        price = player.base_price + int(action) * 2

        if st.session_state.budget >= price:
            st.session_state.budget -= price
            st.session_state.team.append(player.name)

        st.session_state.idx += 1

st.subheader("🏆 Your Team")
st.write(st.session_state.team)

st.subheader("📊 Progress")
st.write(f"Auctioned Players: {st.session_state.idx} / {len(players)}")
