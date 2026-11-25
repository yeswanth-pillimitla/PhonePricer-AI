# -*- coding: utf-8 -*-
"""
Smartphone Price Prediction - Enterprise SaaS Web Application (PhonePricer AI)
Inspired by Linear, Stripe & Vercel design systems.
Features:
- Clean White & Blue Enterprise SaaS aesthetic
- 6 comprehensive interactive pages:
  1. Home & Executive Overview
  2. Interactive Smartphone Price Predictor
  3. Model Benchmarks & Comparison
  4. Feature Explainability & Attribution
  5. Dataset Explorer & Quality Audit
  6. Exploratory Data Analysis (EDA) Gallery
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Ensure workspace root is in python path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import importlib
try:
    import src.predict
    importlib.reload(src.predict)
    from src.predict import predict_smartphone_price, load_model_bundle
except Exception:
    from src.predict import predict_smartphone_price, load_model_bundle

def format_inr(val, symbol: str = "₹") -> str:
    """Format a number into the Indian numbering system (e.g. ₹15,000, ₹1,25,000)."""
    try:
        if val is None:
            return f"{symbol}0"
        val = int(round(float(val)))
    except (ValueError, TypeError):
        return f"{symbol}0"
    is_neg = val < 0
    s = str(abs(val))
    if len(s) <= 3:
        formatted = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.append(rest)
        parts.reverse()
        formatted = ",".join(parts) + "," + last3
    prefix = f"{symbol}" if symbol else ""
    return f"-{prefix}{formatted}" if is_neg else f"{prefix}{formatted}"


# =========================================================================
# PAGE CONFIGURATION & ENTERPRISE SAAS THEME CSS
# =========================================================================
st.set_page_config(
    page_title="PhonePricer AI | Enterprise Price Intelligence",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

CUSTOM_CSS = """
<style>
    /* ===================================================================== */
    /* COLOR SYSTEM & MODERN SAAS TYPOGRAPHY                                  */
    /* ===================================================================== */
    :root {
        --primary-blue: #0878D1;
        --primary-hover: #0660A7;
        --primary-light: #EFF6FF;
        --navy-dark: #102A43;
        --text-main: #172B4D;
        --text-muted: #64748B;
        --bg-main: #F8FAFC;
        --card-bg: #FFFFFF;
        --border-subtle: #E2E8F0;
        --border-accent: #BAE6FD;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }

    .stApp {
        background-color: #F8FAFC !important;
        color: #172B4D !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", Helvetica, Arial, sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #102A43 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* ===================================================================== */
    /* REMOVE LEFT SIDEBAR & CHEVRON COLLAPSE BUTTONS                        */
    /* ===================================================================== */
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"],
    div[data-testid="collapsedControl"],
    button[data-testid="baseButton-headerNoPadding"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        visibility: hidden !important;
    }

    /* Consistent centered responsive container across all pages */
    div[data-testid="stMainBlockContainer"],
    div[data-testid="stAppViewContainer"] section.main .block-container,
    .main .block-container,
    div.block-container {
        max-width: 1360px !important;
        width: 92% !important;
        padding-top: 0.75rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    @media (max-width: 1400px) {
        div[data-testid="stMainBlockContainer"],
        div[data-testid="stAppViewContainer"] section.main .block-container,
        .main .block-container,
        div.block-container {
            max-width: 1280px !important;
            width: 94% !important;
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
        }
    }

    @media (max-width: 1100px) {
        div[data-testid="stMainBlockContainer"],
        div[data-testid="stAppViewContainer"] section.main .block-container,
        .main .block-container,
        div.block-container {
            max-width: 100% !important;
            width: 96% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }

    @media (max-width: 768px) {
        div[data-testid="stMainBlockContainer"],
        div[data-testid="stAppViewContainer"] section.main .block-container,
        .main .block-container,
        div.block-container {
            max-width: 100% !important;
            width: 100% !important;
            padding-top: 0.5rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
    }

    /* Clean Streamlit header */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        z-index: 10 !important;
    }

    /* ===================================================================== */
    /* TOP HORIZONTAL NAVIGATION BAR (SAAS THREE-SECTION DESIGN)              */
    /* ===================================================================== */
    /* Top horizontal block containing the 3 columns (taller, spacious, vertically centered) */
    div[data-testid="stHorizontalBlock"]:has(.st-key-nav_selection) {
        position: sticky !important;
        top: 0.5rem !important;
        z-index: 9999 !important;
        background-color: rgba(255, 255, 255, 0.98) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px -2px rgba(16, 42, 67, 0.08), 0 2px 4px rgba(0, 0, 0, 0.02) !important;
        padding: 10px 18px !important;
        min-height: 62px !important;
        margin-bottom: 24px !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: space-between !important;
        width: 100% !important;
        gap: 8px !important;
        flex-wrap: nowrap !important;
    }

    /* Column 1: Branding (compact, content-width, vertically centered) */
    div[data-testid="stHorizontalBlock"]:has(.st-key-nav_selection) > div[data-testid="column"]:first-child {
        flex: 0 0 auto !important;
        width: auto !important;
        min-width: 160px !important;
        max-width: 185px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
    }

    /* Column 2: Navigation Buttons (centered, isolated, cannot overlap Column 3) */
    div[data-testid="stHorizontalBlock"]:has(.st-key-nav_selection) > div[data-testid="column"]:nth-child(2) {
        flex: 1 1 auto !important;
        width: auto !important;
        min-width: 0 !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        padding: 0 4px !important;
        overflow: hidden !important;
    }

    /* Column 3: Actions & Engine Status (compact, right-aligned, vertically centered) */
    div[data-testid="stHorizontalBlock"]:has(.st-key-nav_selection) > div[data-testid="column"]:last-child {
        flex: 0 0 auto !important;
        width: auto !important;
        min-width: 135px !important;
        max-width: 160px !important;
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        padding: 0 !important;
        margin-left: auto !important;
    }

    /* LEFT SECTION: BRANDING STYLING */
    .topbar-brand-container {
        display: flex;
        align-items: center;
        gap: 10px;
        text-decoration: none;
        user-select: none;
        white-space: nowrap;
    }

    .topbar-logo-avatar {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #0878D1 0%, #102A43 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 6px rgba(8, 120, 209, 0.25);
        flex-shrink: 0;
    }

    .topbar-brand-texts {
        display: flex;
        flex-direction: column;
        justify-content: center;
        line-height: 1.15;
    }

    .topbar-brand-heading {
        font-size: 16px;
        font-weight: 800;
        color: #102A43;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 5px;
    }

    .topbar-ai-pill {
        background-color: #0878D1;
        color: #FFFFFF;
        font-size: 9px;
        font-weight: 700;
        padding: 1px 5px;
        border-radius: 4px;
        letter-spacing: 0.5px;
    }

    .topbar-brand-subheading {
        font-size: 10px;
        color: #64748B;
        font-weight: 500;
        margin-top: 1px;
    }

    /* CENTER SECTION: NAVIGATION WIDGET & RADIOGROUP */
    div.st-key-nav_selection,
    div[data-testid="stRadio"]:has(input[aria-label="NAVIGATION"]) {
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    div.st-key-nav_selection div[role="radiogroup"],
    div.st-key-nav_selection div[data-testid="stRadioGroup"] {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 5px !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        white-space: nowrap !important;
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 2px 4px !important;
    }

    div.st-key-nav_selection div[role="radiogroup"]::-webkit-scrollbar {
        display: none !important;
    }

    /* ===================================================================== */
    /* CIRCULAR RADIO INDICATORS: FULLY VISIBLE, SPACED, AND VERTICALLY ALIGNED */
    /* ===================================================================== */
    /* Ensure the input radio itself is off-screen/zero-sized but accessible */
    div.st-key-nav_selection input[type="radio"] {
        position: absolute !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        pointer-events: none !important;
    }

    /* Base styling for the circular indicator across all options */
    div.st-key-nav_selection div[data-testid="stRadioOption"] label > div:first-child,
    div.st-key-nav_selection div[data-testid="stRadioOption"] [data-baseweb="radio"] > div:first-child,
    div.st-key-nav_selection div[role="radiogroup"] label > div:first-child {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 11px !important;
        min-width: 11px !important;
        max-width: 11px !important;
        height: 11px !important;
        min-height: 11px !important;
        max-height: 11px !important;
        border-radius: 50% !important;
        margin: 0 5px 0 0 !important;
        padding: 0 !important;
        visibility: visible !important;
        opacity: 1 !important;
        flex-shrink: 0 !important;
        border: 1.5px solid #CBD5E1 !important;
        background-color: transparent !important;
        box-sizing: border-box !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    /* Hover state for inactive circular indicator */
    div.st-key-nav_selection div[data-testid="stRadioOption"]:hover label > div:first-child,
    div.st-key-nav_selection div[role="radiogroup"] label:hover > div:first-child {
        border-color: #94A3B8 !important;
        background-color: #F8FAFC !important;
    }

    /* Active circular indicator: crisp red dot matching screenshots */
    div.st-key-nav_selection div[data-testid="stRadioOption"]:has(input:checked) label > div:first-child,
    div.st-key-nav_selection div[data-testid="stRadioOption"]:has(input:checked) [data-baseweb="radio"] > div:first-child,
    div.st-key-nav_selection div[role="radiogroup"] label:has(input:checked) > div:first-child {
        border-color: #EF4444 !important;
        background-color: #EF4444 !important;
        box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.2) !important;
    }

    /* Inner white center dot for active circular indicator */
    div.st-key-nav_selection div[data-testid="stRadioOption"]:has(input:checked) label > div:first-child::after,
    div.st-key-nav_selection div[data-testid="stRadioOption"]:has(input:checked) [data-baseweb="radio"] > div:first-child::after,
    div.st-key-nav_selection div[role="radiogroup"] label:has(input:checked) > div:first-child::after {
        content: "" !important;
        width: 3.5px !important;
        height: 3.5px !important;
        background-color: #FFFFFF !important;
        border-radius: 50% !important;
        display: block !important;
    }

    /* Style navigation button pills (clean typography, no clipping) */
    div.st-key-nav_selection div[role="radiogroup"] label,
    div.st-key-nav_selection div[data-testid="stRadioOption"] {
        background-color: transparent !important;
        padding: 5px 8px !important;
        border-radius: 7px !important;
        cursor: pointer !important;
        transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid transparent !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        margin: 0 !important;
        flex-shrink: 0 !important;
    }

    div.st-key-nav_selection div[role="radiogroup"] label:hover,
    div.st-key-nav_selection div[data-testid="stRadioOption"]:hover {
        background-color: #F1F5F9 !important;
        border-color: #E2E8F0 !important;
        transform: translateY(-1px) !important;
    }

    div.st-key-nav_selection div[role="radiogroup"] label p,
    div.st-key-nav_selection div[role="radiogroup"] label span,
    div.st-key-nav_selection div[data-testid="stRadioOption"] p {
        font-size: 12.5px !important;
        font-weight: 500 !important;
        color: #334155 !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.4 !important;
        letter-spacing: -0.01em !important;
        white-space: nowrap !important;
        transition: color 0.15s ease !important;
    }

    div.st-key-nav_selection div[role="radiogroup"] label:hover p,
    div.st-key-nav_selection div[role="radiogroup"] label:hover span {
        color: #0878D1 !important;
    }

    /* Active navigation button state */
    div.st-key-nav_selection div[role="radiogroup"] label:has(input:checked),
    div.st-key-nav_selection div[data-testid="stRadioOption"]:has(input:checked) {
        background-color: #EAF2FF !important;
        border: 1px solid #BAE6FD !important;
        box-shadow: 0 1px 3px rgba(8, 120, 209, 0.12) !important;
    }

    div.st-key-nav_selection div[role="radiogroup"] label:has(input:checked) p,
    div.st-key-nav_selection div[role="radiogroup"] label:has(input:checked) span {
        color: #0878D1 !important;
        font-weight: 700 !important;
    }

    /* Responsive adjustments for standard laptop screens (<= 1280px) */
    @media (max-width: 1280px) {
        div.st-key-nav_selection div[role="radiogroup"] label,
        div.st-key-nav_selection div[data-testid="stRadioOption"] {
            padding: 4px 6px !important;
        }
        div.st-key-nav_selection div[role="radiogroup"] label p,
        div.st-key-nav_selection div[data-testid="stRadioOption"] p {
            font-size: 12px !important;
        }
        div.st-key-nav_selection div[role="radiogroup"] {
            gap: 2px !important;
        }
    }

    /* RIGHT SECTION: ACTIONS / ENGINE STATUS WIDGET (COMPACT, NO OVERLAP) */
    .topbar-actions-container {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 5px;
        white-space: nowrap;
    }

    .topbar-status-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 2px 7px;
        font-size: 10px;
        color: #102A43;
        white-space: nowrap;
        cursor: default;
        transition: all 0.2s ease;
    }

    .topbar-status-badge:hover {
        border-color: #BAE6FD;
        background-color: #EFF6FF;
    }

    .status-live-pulse {
        width: 6px;
        height: 6px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.25);
    }

    .status-offline-pulse {
        width: 6px;
        height: 6px;
        background-color: #EF4444;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.25);
    }

    .status-engine-name {
        font-weight: 600;
        font-size: 10px;
        color: #102A43;
    }

    .status-live-tag {
        color: #15803D;
        font-weight: 600;
        font-size: 9px;
        background-color: #DCFCE7;
        padding: 1px 4px;
        border-radius: 4px;
        line-height: 1.2;
    }

    .status-offline-tag {
        color: #B91C1C;
        font-weight: 600;
        font-size: 9px;
        background-color: #FEE2E2;
        padding: 1px 4px;
        border-radius: 4px;
        line-height: 1.2;
    }

    .topbar-inr-tag {
        background: #EFF6FF;
        color: #0878D1;
        border: 1px solid #BFDBFE;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 5px;
        display: inline-flex;
        align-items: center;
        white-space: nowrap;
        line-height: 1.2;
    }

    /* Mobile Drawer Popover Styling */
    div[data-testid="stPopover"] {
        display: none !important;
    }

    div[data-testid="stPopover"] button {
        border-radius: 8px !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #F8FAFC !important;
        padding: 4px 10px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #102A43 !important;
    }

    /* Responsive Breakpoints across 1024px, 1280px, 1366px, 1440px, 1920px */
    @media (max-width: 1200px) {
        .topbar-brand-subheading {
            display: none !important;
        }
        div.st-key-nav_selection div[role="radiogroup"] label {
            padding: 6px 9px !important;
        }
        div.st-key-nav_selection div[role="radiogroup"] label p {
            font-size: 12px !important;
        }
    }

    @media (max-width: 960px) {
        .topbar-actions-container {
            display: none !important;
        }
        div[data-testid="stPopover"] {
            display: inline-block !important;
        }
        div.st-key-nav_selection div[role="radiogroup"] {
            justify-content: flex-start !important;
        }
    }

    @media (max-width: 600px) {
        div[data-testid="stHorizontalBlock"]:has(.st-key-nav_selection) {
            padding: 6px 10px !important;
            border-radius: 10px !important;
            margin-bottom: 16px !important;
        }
        div.st-key-nav_selection div[role="radiogroup"] label {
            padding: 5px 8px !important;
        }
        div.st-key-nav_selection div[role="radiogroup"] label p {
            font-size: 11px !important;
        }
    }

    /* ===================================================================== */
    /* HOMEPAGE HERO / WELCOME SECTION                                       */
    /* ===================================================================== */
    .saas-hero-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 28px 32px;
        margin-bottom: 22px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 24px;
        border-top: 4px solid #0878D1;
    }

    .hero-left {
        flex: 1 1 520px;
        min-width: 300px;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #EFF6FF;
        color: #0878D1;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        margin-bottom: 12px;
        border: 1px solid #BFDBFE;
    }

    .hero-badge-dot {
        width: 6px;
        height: 6px;
        background-color: #0878D1;
        border-radius: 50%;
    }

    .hero-title {
        font-size: 32px !important;
        font-weight: 800 !important;
        color: #102A43 !important;
        margin: 0 0 6px 0 !important;
        letter-spacing: -0.03em !important;
        line-height: 1.2 !important;
    }

    .hero-title-accent {
        color: #0878D1;
    }

    .hero-subtitle {
        font-size: 16px;
        font-weight: 600;
        color: #334155;
        margin-bottom: 10px;
    }

    .hero-desc {
        font-size: 14px;
        color: #64748B;
        line-height: 1.55;
        margin: 0;
        max-width: 640px;
    }

    .hero-right {
        flex: 0 1 320px;
        min-width: 260px;
    }

    .hero-glass-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #BFDBFE;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 2px 4px rgba(8, 120, 209, 0.05);
    }

    .glass-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #E2E8F0;
    }

    .glass-tag {
        font-size: 11px;
        font-weight: 700;
        color: #64748B;
        letter-spacing: 0.6px;
    }

    .glass-status {
        font-size: 10px;
        font-weight: 700;
        background-color: #DCFCE7;
        color: #15803D;
        padding: 2px 8px;
        border-radius: 12px;
    }

    .glass-metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 12px;
        margin-bottom: 8px;
    }

    .glass-lbl {
        color: #64748B;
        font-weight: 500;
    }

    .glass-val {
        color: #102A43;
        font-weight: 700;
    }

    /* ===================================================================== */
    /* KPI SUMMARY STAT CARDS                                                */
    /* ===================================================================== */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: all 0.25s ease;
        border-top: 3px solid #0878D1;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .kpi-card:hover {
        box-shadow: 0 6px 12px -2px rgba(8, 120, 209, 0.1);
        transform: translateY(-2px);
        border-color: #BAE6FD;
    }

    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 10px;
    }

    .kpi-label {
        font-size: 11px;
        font-weight: 700;
        color: #64748B;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .kpi-icon-box {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #EFF6FF;
        color: #0878D1;
    }

    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        color: #102A43;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
    }

    .kpi-footer {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }

    .kpi-badge {
        font-size: 11px;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 4px;
    }

    .kpi-badge-blue {
        background-color: #EFF6FF;
        color: #0878D1;
    }

    .kpi-badge-green {
        background-color: #ECFDF5;
        color: #059669;
    }

    .kpi-badge-indigo {
        background-color: #EEF2FF;
        color: #4F46E5;
    }

    .kpi-subtext {
        font-size: 11px;
        color: #64748B;
    }

    /* ===================================================================== */
    /* QUICK ACCESS ACTION CARDS                                             */
    /* ===================================================================== */
    .qa-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: all 0.25s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .qa-card:hover {
        box-shadow: 0 6px 14px -2px rgba(8, 120, 209, 0.12);
        border-color: #BFDBFE;
        transform: translateY(-2px);
    }

    .qa-icon-wrap {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: #EFF6FF;
        color: #0878D1;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        margin-bottom: 12px;
    }

    .qa-title {
        font-size: 15px;
        font-weight: 700;
        color: #102A43;
        margin-bottom: 6px;
    }

    .qa-desc {
        font-size: 12.5px;
        color: #64748B;
        line-height: 1.45;
        margin-bottom: 14px;
        flex-grow: 1;
    }

    /* ===================================================================== */
    /* ML WORKFLOW STAGE CARDS                                               */
    /* ===================================================================== */
    .workflow-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 12px;
        margin: 16px 0;
    }

    .workflow-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        transition: all 0.2s ease;
        position: relative;
    }

    .workflow-card:hover {
        border-color: #93C5FD;
        box-shadow: 0 4px 8px -1px rgba(8, 120, 209, 0.08);
        transform: translateY(-1px);
    }

    .wf-step-num {
        font-size: 11px;
        font-weight: 800;
        color: #0878D1;
        background-color: #EFF6FF;
        padding: 2px 7px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }

    .wf-title {
        font-size: 13.5px;
        font-weight: 700;
        color: #102A43;
        margin-bottom: 4px;
    }

    .wf-detail {
        font-size: 12px;
        color: #64748B;
        line-height: 1.4;
    }

    /* ===================================================================== */
    /* DATA TABLE & PREDICTIONS STYLING                                      */
    /* ===================================================================== */
    .saas-table-container {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin: 14px 0;
    }

    .saas-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        text-align: left;
    }

    .saas-table th {
        background-color: #F8FAFC;
        color: #475569;
        font-weight: 600;
        padding: 12px 16px;
        border-bottom: 1px solid #E2E8F0;
        font-size: 11.5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .saas-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #F1F5F9;
        color: #1E293B;
    }

    .saas-table tr:last-child td {
        border-bottom: none;
    }

    .saas-table tr:hover {
        background-color: #F8FAFC;
    }

    .brand-badge {
        background-color: #EFF6FF;
        color: #0878D1;
        font-weight: 600;
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #BFDBFE;
        display: inline-block;
    }

    .err-pill-green {
        background-color: #DCFCE7;
        color: #15803D;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 12px;
    }

    .err-pill-blue {
        background-color: #E0F2FE;
        color: #0369A1;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 12px;
    }

    .err-pill-amber {
        background-color: #FEF3C7;
        color: #B45309;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 12px;
    }

    /* Section Header Bar */
    .section-header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 28px 0 14px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #E2E8F0;
        flex-wrap: wrap;
        gap: 8px;
    }

    .section-header-title {
        font-size: 18px;
        font-weight: 800;
        color: #102A43;
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 0;
    }

    .section-header-sub {
        font-size: 13px;
        color: #64748B;
        margin-top: 2px;
    }

    /* Google Colab Button styling */
    .colab-action-btn {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background-color: #FFFFFF;
        color: #0878D1 !important;
        font-weight: 700;
        font-size: 13px;
        padding: 8px 16px;
        border-radius: 8px;
        border: 1px solid #BAE6FD;
        text-decoration: none !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
        cursor: pointer;
        white-space: nowrap;
    }
    .colab-action-btn:hover {
        background-color: #F0F9FF;
        color: #0660A7 !important;
        border-color: #38BDF8;
        box-shadow: 0 4px 8px rgba(8, 120, 209, 0.15);
        transform: translateY(-1px);
    }

    /* Download Notebook button styling */
    div[data-testid="stDownloadButton"] button {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        background-color: #FFFFFF !important;
        color: #102A43 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 8px 16px !important;
        border-radius: 8px !important;
        border: 1px solid #BAE6FD !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.2s ease !important;
        height: auto !important;
        min-height: 38px !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background-color: #F0F9FF !important;
        color: #0878D1 !important;
        border-color: #38BDF8 !important;
        box-shadow: 0 4px 8px rgba(8, 120, 209, 0.15) !important;
        transform: translateY(-1px) !important;
    }
    div[data-testid="stDownloadButton"] button [data-testid="stIconMaterial"] {
        color: #0878D1 !important;
        font-size: 18px !important;
    }

    /* Price Banner */
    .price-banner {
        background: linear-gradient(135deg, #0878D1 0%, #102A43 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 15px -3px rgba(8, 120, 209, 0.2);
    }
    .price-val {
        font-size: 38px;
        font-weight: 800;
        margin: 8px 0;
        letter-spacing: -0.5px;
    }
    .price-range {
        font-size: 16px;
        color: #BAE6FD;
        font-weight: 500;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================================
# CACHED DATA & ARTIFACT LOADERS
# =========================================================================
@st.cache_resource
def get_cached_model():
    model_path = os.path.join(WORKSPACE_ROOT, "models", "best_smartphone_price_model.joblib")
    if os.path.exists(model_path):
        return load_model_bundle(model_path)
    return None

@st.cache_data
def get_cleaned_data():
    data_path = os.path.join(WORKSPACE_ROOT, "data", "processed", "cleaned_smartphones.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

@st.cache_data
def get_model_comparison_data():
    comp_path = os.path.join(WORKSPACE_ROOT, "reports", "model_comparison.csv")
    if os.path.exists(comp_path):
        return pd.read_csv(comp_path)
    return None

@st.cache_data
def get_final_predictions_data():
    pred_path = os.path.join(WORKSPACE_ROOT, "reports", "final_test_predictions.csv")
    if os.path.exists(pred_path):
        return pd.read_csv(pred_path)
    return None


bundle = get_cached_model()
df_smartphones = get_cleaned_data()
df_comparison = get_model_comparison_data()
df_test_preds = get_final_predictions_data()


# =========================================================================
# TOP HORIZONTAL NAVIGATION BAR (SAAS THREE-SECTION DESIGN)
# =========================================================================
# Clean navigation labels with emojis removed
NAV_PAGES = [
    "Overview",
    "Predict Smartphone Price",
    "Model Benchmarks",
    "Feature Explainability",
    "Dataset Explorer",
    "EDA Visualization"
]

# Normalization map for backwards-compatibility with old URLs or CTA triggers
PAGE_ALIASES = {
    "🏠 Executive Overview": "Overview",
    "Executive Overview": "Overview",
    "Overview": "Overview",
    "🔮 Predict Smartphone Price": "Predict Smartphone Price",
    "Predict Smartphone Price": "Predict Smartphone Price",
    "🏆 Model Benchmarks": "Model Benchmarks",
    "Model Benchmarks": "Model Benchmarks",
    "🔬 Feature Explainability": "Feature Explainability",
    "Feature Explainability": "Feature Explainability",
    "📊 Dataset Explorer": "Dataset Explorer",
    "Dataset Explorer": "Dataset Explorer",
    "EDA Visualization": "EDA Visualization",
    "EDA Visualizations": "EDA Visualization",
    "📈 EDA Visualizations": "EDA Visualization"
}

if "nav_page" not in st.session_state:
    st.session_state.nav_page = NAV_PAGES[0]
else:
    # Ensure current nav_page is normalized to clean non-emoji name
    st.session_state.nav_page = PAGE_ALIASES.get(st.session_state.nav_page, NAV_PAGES[0])

def on_nav_change():
    selected = st.session_state.nav_selection
    st.session_state.nav_page = PAGE_ALIASES.get(selected, selected)

curr_index = NAV_PAGES.index(st.session_state.nav_page) if st.session_state.nav_page in NAV_PAGES else 0

# Helper function to switch pages programmatically from CTA buttons
def navigate_to(target_page_name):
    clean_target = PAGE_ALIASES.get(target_page_name, target_page_name)
    st.session_state.nav_page = clean_target
    st.session_state.nav_selection = clean_target
    st.rerun()

test_mae = bundle.get("test_metrics", {}).get("Test_MAE", 3831.37) if bundle else 3831.37
test_r2 = bundle.get("test_metrics", {}).get("Test_R2", 0.8513) if bundle else 0.8513
model_name = bundle.get("model_name", "Bounded_Huber") if bundle else "Bounded_Huber"

# Dynamic inference engine readiness status (Online / Offline)
is_model_online = (bundle is not None) and ("pipeline" in bundle or "model" in bundle)
status_tag = "Online" if is_model_online else "Offline"
status_pulse_class = "status-live-pulse" if is_model_online else "status-offline-pulse"
status_tag_class = "status-live-tag" if is_model_online else "status-offline-tag"

# Centered, balanced three-section topbar layout (Brand 11% | Nav 79% | Actions 10%)
with st.container():
    c_brand, c_nav, c_actions = st.columns([1.1, 7.9, 1.0], vertical_alignment="center")
    
    # LEFT SECTION: BRANDING
    with c_brand:
        st.markdown("""
        <div class="topbar-brand-container">
            <div class="topbar-logo-avatar">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="5" y="2" width="14" height="20" rx="2" ry="2"></rect>
                    <line x1="12" y1="18" x2="12.01" y2="18"></line>
                </svg>
            </div>
            <div class="topbar-brand-texts">
                <div class="topbar-brand-heading">PhonePricer <span class="topbar-ai-pill">AI</span></div>
                <div class="topbar-brand-subheading">Enterprise Price Intelligence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # CENTER SECTION: NAVIGATION (Overview completely visible, no emojis, spacious)
    with c_nav:
        nav_page = st.radio(
            "NAVIGATION",
            NAV_PAGES,
            index=curr_index,
            horizontal=True,
            label_visibility="collapsed",
            key="nav_selection",
            on_change=on_nav_change
        )
        st.session_state.nav_page = PAGE_ALIASES.get(nav_page, nav_page)

    # RIGHT SECTION: ACTIONS / PRODUCTION ENGINE STATUS & MOBILE MENU (COMPACT, NO OVERLAP)
    with c_actions:
        st.markdown(f"""
        <div class="topbar-actions-container">
            <div class="topbar-status-badge" title="Trained Architecture: {model_name} | Status: {status_tag} | Partition: 70/15/15 | Test MAE: ±{format_inr(test_mae)}">
                <span class="{status_pulse_class}"></span>
                <span class="status-engine-name">{model_name}</span>
                <span class="{status_tag_class}">{status_tag}</span>
            </div>
            <div class="topbar-inr-tag" title="Valuation currency strictly Indian Rupees">
                ₹ INR
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.popover("☰ Menu", help="Mobile Navigation Drawer"):
            st.markdown("**Navigation Menu**")
            for page in NAV_PAGES:
                if st.button(page, key=f"mob_btn_{page}", use_container_width=True):
                    navigate_to(page)


# =========================================================================
# PAGE 1: EXECUTIVE OVERVIEW (ENTERPRISE SAAS HOMEPAGE REDESIGN)
# =========================================================================
if nav_page in ["Overview", "🏠 Executive Overview"]:
    # -------------------------------------------------------------
    # 1. HERO / WELCOME SECTION
    # -------------------------------------------------------------
    test_mae = bundle.get("test_metrics", {}).get("Test_MAE", 3831.37) if bundle else 3831.37
    test_r2 = bundle.get("test_metrics", {}).get("Test_R2", 0.8513) if bundle else 0.8513
    model_name = bundle.get("model_name", "Bounded_Huber") if bundle else "Bounded_Huber"
    
    st.markdown(f"""
    <div class="saas-hero-container">
        <div class="hero-left">
            <div class="hero-badge">
                <span class="hero-badge-dot"></span>
                <span>Enterprise Machine Learning Platform</span>
            </div>
            <h1 class="hero-title">PhonePricer <span class="hero-title-accent">AI</span></h1>
            <div class="hero-subtitle">Intelligent Smartphone Price Prediction Powered by Machine Learning</div>
            <p class="hero-desc">
                Predict smartphone market prices using hardware specifications and advanced machine learning algorithms.
                Engineered with strict zero-leakage preprocessing, 16 benchmarked architectures, and monotonic pricing bounds.
            </p>
        </div>
        <div class="hero-right">
            <div class="hero-glass-card">
                <div class="glass-header">
                    <span class="glass-tag">SYSTEM METRICS</span>
                    <span class="glass-status">Production Ready</span>
                </div>
                <div class="glass-metric-row">
                    <span class="glass-lbl">Active Architecture</span>
                    <span class="glass-val">{model_name}</span>
                </div>
                <div class="glass-metric-row">
                    <span class="glass-lbl">Generalization Fit</span>
                    <span class="glass-val">{test_r2*100:.1f}% Variance Explained</span>
                </div>
                <div class="glass-metric-row">
                    <span class="glass-lbl">Valuation Currency</span>
                    <span class="glass-val">Indian Rupee (₹ INR)</span>
                </div>
                <div class="glass-metric-row">
                    <span class="glass-lbl">Dataset Diversity</span>
                    <span class="glass-val">91 Devices • 11 Global Brands</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # -------------------------------------------------------------
    # 2. KEY PERFORMANCE INDICATORS (KPI CARDS)
    # -------------------------------------------------------------
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">Total Dataset Records</span>
                <div class="kpi-icon-box">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0878D1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
                        <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>
                        <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>
                    </svg>
                </div>
            </div>
            <div class="kpi-value">91</div>
            <div class="kpi-footer">
                <span class="kpi-badge kpi-badge-blue">11 Brands</span>
                <span class="kpi-subtext">Verified Specifications</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">Best Performing Model</span>
                <div class="kpi-icon-box">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0878D1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="8" r="7"></circle>
                        <polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"></polyline>
                    </svg>
                </div>
            </div>
            <div class="kpi-value" style="font-size: 22px; line-height: 34px;">{model_name}</div>
            <div class="kpi-footer">
                <span class="kpi-badge kpi-badge-blue">Tuned Monotonic</span>
                <span class="kpi-subtext">Selected via 5-Fold CV</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">Held-out Test MAE</span>
                <div class="kpi-icon-box">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <circle cx="12" cy="12" r="6"></circle>
                        <circle cx="12" cy="12" r="2"></circle>
                    </svg>
                </div>
            </div>
            <div class="kpi-value">±{format_inr(test_mae)}</div>
            <div class="kpi-footer">
                <span class="kpi-badge kpi-badge-green">Unbiased Test</span>
                <span class="kpi-subtext">Held-out Error Margin</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">Test R² Score</span>
                <div class="kpi-icon-box">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6366F1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="18" y1="20" x2="18" y2="10"></line>
                        <line x1="12" y1="20" x2="12" y2="4"></line>
                        <line x1="6" y1="20" x2="6" y2="14"></line>
                    </svg>
                </div>
            </div>
            <div class="kpi-value">{test_r2*100:.1f}%</div>
            <div class="kpi-footer">
                <span class="kpi-badge kpi-badge-indigo">High Correlation</span>
                <span class="kpi-subtext">Unseen Evaluation Fit</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 3. ML WORKFLOW VISUALIZATION (Replacing Long Text)
    # -------------------------------------------------------------
    st.markdown("""
    <div class="section-header-bar">
        <div>
            <h2 class="section-header-title">🔄 End-to-End Machine Learning Pipeline</h2>
            <div class="section-header-sub">An 8-stage enterprise data science workflow ensuring zero data leakage and physical validity.</div>
        </div>
    </div>
    
    <div class="workflow-grid">
        <div class="workflow-card">
            <span class="wf-step-num">STAGE 01</span>
            <div class="wf-title">📥 Dataset Collection</div>
            <div class="wf-detail">91 authentic smartphone models across 11 global brands with 30 raw technical attributes.</div>
        </div>
        <div class="workflow-card">
            <span class="wf-step-num">STAGE 02</span>
            <div class="wf-title">🛡️ Data Cleaning & Audit</div>
            <div class="wf-detail">Zero-variance verification, structural zero handling for CPU clusters, and physical bounds validation.</div>
        </div>
        <div class="workflow-card">
            <span class="wf-step-num">STAGE 03</span>
            <div class="wf-title">⚙️ Feature Engineering</div>
            <div class="wf-detail">Derived 44 hardware features including Silicon Density, Display Bandwidth, and CPU cluster topologies.</div>
        </div>
        <div class="workflow-card">
            <span class="wf-step-num">STAGE 04</span>
            <div class="wf-title">🔄 Preprocessing Pipeline</div>
            <div class="wf-detail">Strict anti-leakage ColumnTransformer with RobustScaler and OneHotEncoder fitted strictly on Train partition.</div>
        </div>
        <div class="workflow-card">
            <span class="wf-step-num">STAGE 05</span>
            <div class="wf-title">📊 Stratified Splitting</div>
            <div class="wf-detail">70% Train (63), 15% Val (14), 15% Test (14) with 5-quantile price stratification across budget & flagships.</div>
        </div>
        <div class="workflow-card">
            <span class="wf-step-num">STAGE 06</span>
            <div class="wf-title">🧠 Multi-Model Training</div>
            <div class="wf-detail">Evaluated 16 regression algorithms (Ridge, Lasso, Huber, SVR, Trees, Ensembles, and XGBoost).</div>
        </div>
        <div class="workflow-card">
            <span class="wf-step-num">STAGE 07</span>
            <div class="wf-title">🎯 Model Evaluation</div>
            <div class="wf-detail">Repeated 5-Fold CV tuning with GridSearchCV, overfitting gap control, and validation holdout comparison.</div>
        </div>
        <div class="workflow-card">
            <span class="wf-step-num">STAGE 08</span>
            <div class="wf-title">💰 Price Prediction</div>
            <div class="wf-detail">Monotonic bounded inference delivering valuations strictly in Indian Rupees (₹) with error brackets.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📖 View Full Technical & Mathematical Architecture Documentation", expanded=False):
        st.markdown("""
        ### Comprehensive Architectural Methodology
        
        1. **Data Quality & Physical Validation**:
           - Verified non-zero positive prices across all records.
           - Confirmed that multi-core counts sum exactly to `Processor_Core_Count` (8 cores).
           - Checked structural zeros: Dual-cluster octa-cores (e.g. 4+4) have 0 count and 0 GHz in Cluster 3 by hardware definition.
           - Validated physical ranges: Process technology (3nm – 28nm), RAM (2GB – 24GB), Battery (4,000mAh – 7,000mAh).
           
        2. **Anti-Leakage Preprocessing Pipeline**:
           - Implemented via Scikit-Learn `ColumnTransformer` embedded inside an isolated `Pipeline`.
           - **Numerical Features**: Transformed using `RobustScaler` (median-centered and IQR-scaled to resist extreme flagship pricing spikes).
           - **Categorical Features**: Encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to safely ingest unseen phone brands.
           - Preprocessing parameters (μ, σ, IQR) are fitted **solely on $X_{train}$** and transferred freeze-fitted to validation and test sets.
           
        3. **Stratified Splitting Protocol**:
           - Applied 5-quantile target price binning (`pd.qcut`) to ensure balanced distribution of budget (₹7k–₹15k), mid-tier (₹15k–₹35k), and flagship (₹35k–₹75k) devices across partitions.
           - Exact counts: 63 Train samples, 14 Validation samples, 14 Test samples.
           
        4. **Model Selection & Monotonicity Constraint**:
           - Ordinary Least Squares regression demonstrated extreme overfitting due to multicollinearity across 44 one-hot encoded dimensions.
           - Regularized linear and robust architectures (`Ridge`, `Bounded_Huber`) achieved superior bias-variance balance.
           - Selected **`Bounded_Huber`** with non-negative battery constraint, guaranteeing that increasing battery capacity consistently yields non-decreasing price valuations.
        """)

    # -------------------------------------------------------------
    # 5. MODEL PERFORMANCE OVERVIEW
    # -------------------------------------------------------------
    st.markdown("""
    <div class="section-header-bar">
        <div>
            <h2 class="section-header-title">📈 Model Performance Overview</h2>
            <div class="section-header-sub">Empirical validation metrics across algorithms and held-out test predictions.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    perf_col1, perf_col2 = st.columns([1.2, 1])
    with perf_col1:
        st.subheader("Top Models by Validation MAE (Lower is Better)")
        if df_comparison is not None:
            # Sort by Val_MAE and display top 6
            top_models_df = df_comparison.sort_values(by="Val_MAE", ascending=True).head(6)
            chart_data = top_models_df.set_index("Model_Name")[["Val_MAE"]]
            chart_data.columns = ["Validation MAE (₹)"]
            st.bar_chart(chart_data, color="#0878D1")
            
            # Key highlight metrics
            hl_c1, hl_c2, hl_c3 = st.columns(3)
            hl_c1.metric("Winning Model", model_name, "Lowest Error")
            hl_c2.metric("Test Explained Fit", f"{test_r2*100:.1f}%", "+0.851 R²")
            hl_c3.metric("Test MAE Margin", f"±{format_inr(test_mae)}", "Unbiased")
        else:
            st.info("Model benchmark data is loading...")
            
    with perf_col2:
        st.subheader("Actual vs Predicted Parity Plot (Test Set)")
        parity_fig_path = os.path.join(WORKSPACE_ROOT, "reports", "figures", "actual_vs_predicted.png")
        if os.path.exists(parity_fig_path):
            st.image(parity_fig_path, caption="Held-Out Test Set (14 Unseen Phones) • Parity Line (y = x)", use_container_width=True)
        else:
            st.info("Parity chart artifact will be displayed once generated.")

    # -------------------------------------------------------------
    # 6. DATASET OVERVIEW
    # -------------------------------------------------------------
    st.markdown("""
    <div class="section-header-bar">
        <div>
            <h2 class="section-header-title">📊 Dataset Specifications & Brand Coverage</h2>
            <div class="section-header-sub">Comprehensive technical coverage across modern mobile platforms and silicon architectures.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    ds_c1, ds_c2, ds_c3 = st.columns(3)
    with ds_c1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">11 GLOBAL BRANDS</span>
                <div class="kpi-icon-box">🏷️</div>
            </div>
            <div style="font-size: 13px; color: #334155; line-height: 1.7; margin-top: 4px;">
                <span class="brand-badge">Xiaomi Redmi</span> <span class="brand-badge">OnePlus</span> <span class="brand-badge">Samsung</span>
                <span class="brand-badge">Vivo</span> <span class="brand-badge">Oppo</span> <span class="brand-badge">POCO</span>
                <span class="brand-badge">Motorola</span> <span class="brand-badge">Realme</span> <span class="brand-badge">Infinix</span>
                <span class="brand-badge">Nokia</span> <span class="brand-badge">Nothing</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with ds_c2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">SILICON & HARDWARE COVERAGE</span>
                <div class="kpi-icon-box">⚡</div>
            </div>
            <div style="font-size: 12.5px; color: #475569; line-height: 1.6; margin-top: 4px;">
                • <strong>Lithography</strong>: 3nm, 4nm, 6nm, 12nm, 28nm<br>
                • <strong>RAM Tiers</strong>: 2GB, 3GB, 4GB, 6GB, 8GB, 12GB, 16GB, 24GB<br>
                • <strong>Storage Tiers</strong>: 32GB, 64GB, 128GB, 256GB, 512GB, 1TB<br>
                • <strong>Batteries</strong>: 4,000 mAh up to 7,000 mAh
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with ds_c3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">RETAIL PRICE RANGE (INR)</span>
                <div class="kpi-icon-box">₹</div>
            </div>
            <div style="font-size: 24px; font-weight: 800; color: #0878D1; margin: 4px 0;">
                ₹7,550 – ₹75,494
            </div>
            <div style="font-size: 12px; color: #64748B;">
                Balanced representation from entry-level Android Go devices to top-tier flagship phones in India.
            </div>
        </div>
        """, unsafe_allow_html=True)



# ==========================================
# PAGE 2: PRICE PREDICTOR
# ==========================================
elif nav_page in ["Predict Smartphone Price", "🔮 Predict Smartphone Price"]:
    colab_notebook_path = os.path.join(WORKSPACE_ROOT, "notebooks", "PhonePricer_AI_Graphs_Fixed.ipynb")
    if not os.path.exists(colab_notebook_path):
        colab_notebook_path = os.path.join(WORKSPACE_ROOT, "notebooks", "PhonePricer_AI.ipynb")
    
    # Configure GitHub repository detection
    default_github_repo = "yeswanth-pillimitla/PhonePricer-AI"
    github_colab_url = f"https://colab.research.google.com/github/{default_github_repo}/blob/main/notebooks/PhonePricer_AI_Graphs_Fixed.ipynb"
    upload_colab_url = "https://colab.research.google.com/#upload=true"
    
    # Check whether the GitHub repository exists live on GitHub
    is_github_live = False
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://api.github.com/repos/{default_github_repo}/contents/notebooks/PhonePricer_AI_Graphs_Fixed.ipynb",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                is_github_live = True
    except Exception:
        is_github_live = False

    # Prevent 404: Use direct GitHub link if live, otherwise use native Colab Upload interface
    primary_colab_url = github_colab_url if is_github_live else upload_colab_url
    colab_btn_tooltip = f"Direct GitHub Colab Link ({default_github_repo})" if is_github_live else "Open Google Colab (Upload interface - zero 404 errors)"

    st.markdown(f"""
    <div class="main-header" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
        <div style="flex: 1; min-width: 280px;">
            <h1 style="margin: 0; color: #102A43; font-size: 28px; font-weight: 700;">🔮 Custom Smartphone Price Predictor</h1>
            <p style="margin: 6px 0 0 0; color: #000000; font-size: 14.5px; line-height: 1.5;">Configure hardware specifications or select a preset to compute real-time market price valuation in Indian Rupees (₹).</p>
        </div>
        <div>
            <a href="{primary_colab_url}" target="_blank" class="colab-action-btn" title="{colab_btn_tooltip}">
                <svg width="18" height="18" viewBox="-11.5 -10.23174 23 20.46348" fill="none">
                    <circle cx="0" cy="0" r="2.05" fill="#0878D1"/>
                    <g stroke="#0878D1" stroke-width="1.1">
                        <ellipse rx="11" ry="4.2"/>
                        <ellipse rx="11" ry="4.2" transform="rotate(60)"/>
                        <ellipse rx="11" ry="4.2" transform="rotate(120)"/>
                    </g>
                </svg>
                <span>Open in Google Colab</span>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="opacity: 0.7;">
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                    <polyline points="15 3 21 3 21 9"></polyline>
                    <line x1="10" y1="14" x2="21" y2="3"></line>
                </svg>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Colab Export & Interactive Offline Guide
    if os.path.exists(colab_notebook_path):
        with open(colab_notebook_path, "rb") as nb_file:
            colab_bytes = nb_file.read()
        col_nb1, col_nb2 = st.columns([2.8, 1.2])
        with col_nb2:
            st.download_button(
                label="Download Notebook (.ipynb)",
                icon=":material/download:",
                data=colab_bytes,
                file_name="PhonePricer_AI_Graphs_Fixed.ipynb",
                mime="application/x-ipynb+json",
                use_container_width=True
            )

        with st.expander("ℹ️ How to run in Google Colab (2 Easy Options)", expanded=False):
            st.markdown(f"""
            #### 🔍 Why did Colab previously show a 404 error?
            Google Colab's GitHub fetcher searches for a **public repository on GitHub**. If the project has not yet been pushed to GitHub under `yeswanth-pillimitla/PhonePricer-AI`, Colab cannot find the file and returns `404: Not Found`.

            ---
            
            #### 🚀 Option 1: Instant Run in Colab (Works immediately — No GitHub needed!)
            1. Click **Download Notebook (.ipynb)** above to save `PhonePricer_AI_Graphs_Fixed.ipynb`.
            2. Click **Open in Google Colab** above (opens Google Colab directly to the **Upload** tab).
            3. Select or drag-and-drop the downloaded `PhonePricer_AI_Graphs_Fixed.ipynb` file into the upload box.
            4. In **Cell 4**, upload `Coa-12.csv` and run all cells (`Runtime -> Run all` or `Ctrl + F9`).

            ---

            #### 🌐 Option 2: 1-Click Direct GitHub Integration
            To enable direct 1-click opening from GitHub without downloading:
            1. Create a public repository named **`PhonePricer-AI`** on [GitHub](https://github.com/new).
            2. In your local terminal, link and push your repository (it is already committed locally):
               ```bash
               git remote add origin https://github.com/yeswanth-pillimitla/PhonePricer-AI.git
               git push -u origin main
               ```
            3. Refresh this page. The button will automatically detect your live repo and open it directly from GitHub!
            """)
    
    # SoC (System on Chip) Selector with all unique SoCs from dataset
    soc_options = ["Custom Specification"]
    if df_smartphones is not None and "SoC" in df_smartphones.columns:
        soc_options += sorted(df_smartphones["SoC"].dropna().unique().tolist())
        
    selected_soc = st.selectbox(
        "SoC (System on Chip)",
        soc_options,
        help="Select an authentic System on Chip (SoC) from the dataset to auto-populate hardware specifications, or choose 'Custom Specification' to enter manually."
    )
    
    # Pre-populate specifications based on selected SoC or default custom values
    if selected_soc != "Custom Specification" and df_smartphones is not None:
        match = df_smartphones[df_smartphones["SoC"] == selected_soc].iloc[0]
        d_brand = match["Phone_Company"]
        d_ram = int(match["RAM_Size_GB"])
        d_ram_t = match["RAM_Type"]
        d_stor = int(match["Storage_GB"])
        d_stor_max = int(match["Storage_GB_Max"]) if "Storage_GB_Max" in match else int(match["Storage_GB"])
        d_bat = int(match["Battery_Capacity_mAh"])
        d_nm = int(match["Process_Technology_nm"])
        d_w = int(match["Display_Resolution_Width_px"])
        d_h = int(match["Display_Resolution_Height_px"])
        d_hz = int(match["Refresh_Rate_Hz"])
        d_dtype = match["Display_Type"]
        d_c1_c = int(match["Processor_Core1_Count"])
        d_c1_f = float(match["Processor_Core1_Frequency_GHz"])
        d_c2_c = int(match["Processor_Core2_Count"])
        d_c2_f = float(match["Processor_Core2_Frequency_GHz"])
        d_c3_c = int(match["Processor_Core3_Count"])
        d_c3_f = float(match["Processor_Core3_Frequency_GHz"])
        d_c1_cache = int(match["Cache_Memory_Core1"])
        d_c2_cache = int(match["Cache_Memory_Core2"])
        d_c3_cache = int(match["Cache_Memory_Core3"])
        d_isa = match["ISA"]
        d_cam = match["Camera_Specifications"]
        from src.feature_engineering import extract_rear_camera_mp
        d_rear_mp = extract_rear_camera_mp(d_cam)
    else:
        d_brand, d_ram, d_ram_t, d_stor, d_stor_max = "Samsung", 8, "LPDDR4X", 128, 256
        d_bat, d_nm, d_w, d_h, d_hz, d_dtype = 5000, 4, 1080, 2340, 120, "Super AMOLED, 120Hz"
        d_c1_c, d_c1_f, d_c2_c, d_c2_f, d_c3_c, d_c3_f = 4, 2.4, 4, 1.8, 0, 0.0
        d_c1_cache, d_c2_cache, d_c3_cache, d_isa, d_cam = 512, 128, 0, "Armv8.2-A", "Rear: 50 MP, OIS | Front: 13 MP"
        d_rear_mp = 50.0

    with st.form("prediction_form"):
        st.subheader("1. Manufacturer & Platform Architecture")
        colA, colB, colC, colD = st.columns(4)
        with colA:
            brand_opts = ["Xiaomi Redmi", "POCO", "Infinix", "Vivo", "Oppo", "OnePlus", "Samsung", "Motorola", "Realme", "Nokia", "Nothing", "Other Brand"]
            brand_idx = brand_opts.index(d_brand) if d_brand in brand_opts else 0
            phone_company = st.selectbox("Smartphone Brand", brand_opts, index=brand_idx)
        with colB:
            process_nm = st.slider("Process Lithography (nm)", min_value=3, max_value=28, value=d_nm, help="Smaller nm indicates more advanced, efficient silicon.")
        with colC:
            isa_opts = ["Armv8-A", "Armv8.2-A", "Armv9-A", "Armv9.2-A"]
            isa_idx = isa_opts.index(d_isa) if d_isa in isa_opts else 1
            isa = st.selectbox("Instruction Set Architecture (ISA)", isa_opts, index=isa_idx)
        with colD:
            rt_opts = ["LPDDR3", "LPDDR4X", "LPDDR5", "LPDDR5X"]
            rt_idx = rt_opts.index(d_ram_t) if d_ram_t in rt_opts else 1
            ram_type = st.selectbox("RAM Generation", rt_opts, index=rt_idx)

        st.subheader("2. Memory, Storage & Battery Capacity")
        colE, colF, colG = st.columns(3)
        with colE:
            ram_size = st.select_slider("Base RAM (GB)", options=[2, 3, 4, 6, 8, 12, 16, 24], value=d_ram if d_ram in [2, 3, 4, 6, 8, 12, 16, 24] else 8)
        with colF:
            storage_size = st.select_slider("Base Storage (GB)", options=[16, 32, 64, 128, 256, 512], value=d_stor if d_stor in [16, 32, 64, 128, 256, 512] else 128)
        with colG:
            battery_mah = st.number_input("Battery Capacity (mAh)", min_value=2500, max_value=9000, value=d_bat, step=100)

        st.subheader("3. Processor Core Clustering & Clock Speeds (Total 8 Cores)")
        st.caption("Octa-core layouts: either Dual-Cluster (Core1 + Core2 = 8) or Tri-Cluster (Core1 + Core2 + Core3 = 8). Set Core3 to 0 for Dual-Cluster.")
        colI, colJ, colK = st.columns(3)
        with colI:
            st.markdown("**Cluster 1 (Prime / Performance)**")
            c1_count = st.number_input("Core 1 Count", min_value=1, max_value=4, value=d_c1_c)
            c1_freq = st.number_input("Core 1 Clock (GHz)", min_value=1.0, max_value=5.0, value=float(d_c1_f), step=0.1)
            c1_cache_opts = [128, 256, 512, 1024, 2048, 8192, 12288]
            c1_cache_idx = c1_cache_opts.index(d_c1_cache) if d_c1_cache in c1_cache_opts else 2
            c1_cache = st.selectbox("Core 1 Cache (KB)", c1_cache_opts, index=c1_cache_idx)
        with colJ:
            st.markdown("**Cluster 2 (Performance / Efficiency)**")
            c2_count = st.number_input("Core 2 Count", min_value=1, max_value=6, value=d_c2_c)
            c2_freq = st.number_input("Core 2 Clock (GHz)", min_value=1.0, max_value=4.0, value=float(d_c2_f), step=0.05)
            c2_cache_opts = [128, 256, 512, 1024, 8192, 12288]
            c2_cache_idx = c2_cache_opts.index(d_c2_cache) if d_c2_cache in c2_cache_opts else 1
            c2_cache = st.selectbox("Core 2 Cache (KB)", c2_cache_opts, index=c2_cache_idx)
        with colK:
            st.markdown("**Cluster 3 (Efficiency - Structural Zeros if Dual-Cluster)**")
            c3_count = st.number_input("Core 3 Count", min_value=0, max_value=6, value=d_c3_c)
            c3_freq = st.number_input("Core 3 Clock (GHz)", min_value=0.0, max_value=3.0, value=float(d_c3_f), step=0.1)
            c3_cache_opts = [0, 128, 256, 512, 1024]
            c3_cache_idx = c3_cache_opts.index(d_c3_cache) if d_c3_cache in c3_cache_opts else 0
            c3_cache = st.selectbox("Core 3 Cache (KB)", c3_cache_opts, index=c3_cache_idx)

        st.subheader("4. Display Panel & Primary Camera Sensor")
        colL, colM, colN, colO = st.columns(4)
        with colL:
            dtype_opts = ["AMOLED, 120Hz", "AMOLED, 144Hz", "IPS LCD, 90Hz", "IPS LCD, 120Hz", "Super AMOLED, 120Hz", "OLED, 120Hz"]
            if d_dtype not in dtype_opts:
                dtype_opts.insert(0, d_dtype)
            dtype_idx = dtype_opts.index(d_dtype) if d_dtype in dtype_opts else 0
            display_type = st.selectbox("Display Technology", dtype_opts, index=dtype_idx)
        with colM:
            res_w_opts = [720, 1080, 1156, 1208, 1220, 1224, 1260, 1264, 1272, 1280, 1440]
            res_w_idx = res_w_opts.index(d_w) if d_w in res_w_opts else 1
            res_w = st.selectbox("Horizontal Resolution (px)", res_w_opts, index=res_w_idx)
        with colN:
            res_h = st.number_input("Vertical Resolution (px)", min_value=1400, max_value=3400, value=d_h, step=20)
        with colO:
            hz_opts = [60, 90, 120, 144, 165, 185]
            hz_idx = hz_opts.index(d_hz) if d_hz in hz_opts else 2
            refresh_hz = st.selectbox("Refresh Rate (Hz)", hz_opts, index=hz_idx)

        colP, colQ = st.columns(2)
        with colP:
            rear_mp_opts = [8.0, 13.0, 32.0, 48.0, 50.0, 64.0, 108.0, 200.0]
            rear_mp_idx = rear_mp_opts.index(d_rear_mp) if d_rear_mp in rear_mp_opts else 4
            rear_mp = st.selectbox("Primary Rear Camera (MP)", rear_mp_opts, index=rear_mp_idx)
        with colQ:
            storage_max = st.select_slider("Max Storage Tier (GB)", options=[32, 64, 128, 256, 512, 1024], value=d_stor_max if d_stor_max in [32, 64, 128, 256, 512, 1024] else 256)

        submitted = st.form_submit_button("🚀 Compute Predicted Market Price", use_container_width=True)

    if submitted:
        # Physical validation check
        total_cores = c1_count + c2_count + c3_count
        if total_cores != 8:
            st.warning(f"⚠️ Notice: Core sum is {total_cores} cores. Typical mobile octa-cores total 8. Adjusting automatically.")
            
        spec_dict = {
            "SoC": selected_soc if selected_soc != "Custom Specification" else "Standard Octa-Core SoC",
            "Phone_Company": phone_company,
            "RAM_Type": ram_type,
            "RAM_Size_GB": ram_size,
            "Storage_GB": storage_size,
            "Battery_Capacity_mAh": battery_mah,
            "Process_Technology_nm": process_nm,
            "Display_Resolution_Width_px": res_w,
            "Display_Resolution_Height_px": res_h,
            "Refresh_Rate_Hz": refresh_hz,
            "Display_Type": display_type,
            "Processor_Core1_Count": c1_count,
            "Processor_Core1_Frequency_GHz": c1_freq,
            "Processor_Core2_Count": c2_count,
            "Processor_Core2_Frequency_GHz": c2_freq,
            "Processor_Core3_Count": c3_count,
            "Processor_Core3_Frequency_GHz": c3_freq,
            "Cache_Memory_Core1": c1_cache,
            "Cache_Memory_Core2": c2_cache,
            "Cache_Memory_Core3": c3_cache,
            "ISA": isa,
            "Camera_Specifications": f"Rear: {rear_mp} MP"
        }
        
        result = predict_smartphone_price(spec_dict)
        
        st.markdown(f"""
        <div class="price-banner">
            <div style="text-transform: uppercase; letter-spacing: 1px; font-size: 13px; font-weight: 600;">Valuation Forecast</div>
            <div class="price-val">{result['formatted_price']}</div>
            <div class="price-range">Expected Valuation Bracket: <strong>{result['formatted_range']}</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.info(f"💡 **Model Confidence & Evaluation Limitations**: {result['confidence_note']}")
        
        # Display derived breakdown
        st.markdown("### Derived Architecture Specifications")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        col_d1.metric("Peak Core Clock", f"{max(c1_freq, c2_freq, c3_freq):.2f} GHz")
        col_d2.metric("Total Screen Pixels", f"{(res_w * res_h) / 1e6:.2f} MP")
        col_d3.metric("CPU Clusters", "3-Cluster (Tri)" if c3_count > 0 else "2-Cluster (Dual)")
        col_d4.metric("Aspect Ratio", f"{res_h / res_w:.2f}:1")


# ==========================================
# PAGE 3: MODEL BENCHMARKS
# ==========================================
elif nav_page in ["Model Benchmarks", "🏆 Model Benchmarks"]:
    st.markdown("""
    <div class="main-header">
        <h1>🏆 Model Benchmark & Comparative Analysis</h1>
        <p>Rigorous multi-metric evaluation across Dummy baselines, Linear, Regularized, Tree Ensembles, Support Vector, and Instance-Based algorithms.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if df_comparison is not None:
        st.subheader("Model Leaderboard (Sorted by Validation MAE)")
        st.dataframe(
            df_comparison[[
                "Model_Name", "Family", "Val_MAE", "Val_RMSE", "Val_R2",
                "CV_MAE_Mean", "CV_MAE_Std", "Overfitting_Gap_MAE"
            ]],
            use_container_width=True
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("Validation MAE Comparison (Lower is Better)")
            st.bar_chart(data=df_comparison.set_index("Model_Name")["Val_MAE"])
        with col_g2:
            st.subheader("Validation R² Score (Higher is Better)")
            st.bar_chart(data=df_comparison.set_index("Model_Name")["Val_R2"])
            
        st.markdown("""
        ### Key Scientific Insights from Benchmarks:
        1. **Baselines Outperformed**: The Dummy Regressors (Mean and Median) recorded validation MAEs of ~₹15,000+ with $R^2 \\le 0$. Every tuned model substantially surpassed the baseline.
        2. **Tree Ensembles vs. Regularized Linear**:
           - **Tuned Ensembles & Bounded Huber** achieved low Validation MAE and high $R^2$ with an overfitting gap under ₹800.
           - **ElasticNet** and **Ridge** demonstrated outstanding regularized linear stability, keeping overfitting gaps nearly neutral ($< ₹300$).
        3. **Unregularized Linear Failure**: Ordinary least squares linear regression suffered from extreme variance ($R^2 < 0$, CV MAE ~₹25,000) due to multicollinearity across 44 one-hot encoded features on 63 training instances.
        """)


# ==========================================
# PAGE 4: FEATURE EXPLAINABILITY
# ==========================================
elif nav_page in ["Feature Explainability", "🔬 Feature Explainability"]:
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Model Explainability & Feature Attribution</h1>
        <p>Exposing the physical hardware specifications that drive smartphone retail prices.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Tree MDI Importance", "Permutation Importance", "Linear Beta Weights"])
    
    with tab1:
        st.subheader("Random Forest Impurity Importance (Mean Decrease in Impurity)")
        fig_path = os.path.join(WORKSPACE_ROOT, "reports", "figures", "feature_importance.png")
        if os.path.exists(fig_path):
            st.image(fig_path, caption="Figure: Top Gini feature importance scores across tree splits.", use_container_width=True)
        else:
            st.warning("Figure not generated yet.")
            
    with tab2:
        st.subheader("Model-Agnostic Permutation Importance (Validation Partition)")
        perm_path = os.path.join(WORKSPACE_ROOT, "reports", "figures", "permutation_importance.png")
        if os.path.exists(perm_path):
            st.image(perm_path, caption="Figure: Permutation Importance (Degradation in MAE upon feature shuffling).", use_container_width=True)
        else:
            st.warning("Figure not generated yet.")
            
    with tab3:
        st.subheader("Regularized Linear Model Standardized Weights (Beta Coefficients)")
        coef_path = os.path.join(WORKSPACE_ROOT, "reports", "figures", "linear_coefficients.png")
        if os.path.exists(coef_path):
            st.image(coef_path, caption="Figure: Relative positive (blue) and negative (red) pricing pressures.", use_container_width=True)
        else:
            st.warning("Figure not generated yet.")
            
    st.markdown("""
    > **Methodological Note on Collinearity**:
    > In commercial smartphones, high-performance silicon (3nm nodes, 3.2GHz+ clock speeds) is bundled systematically with large LPDDR5X RAM and high-resolution AMOLED displays.
    > Attribution scores reflect *predictive association* across the current market packaging rather than isolated causal markups.
    """)


# ==========================================
# PAGE 5: DATASET EXPLORER
# ==========================================
elif nav_page in ["Dataset Explorer", "📊 Dataset Explorer"]:
    st.markdown("""
    <div class="main-header">
        <h1>📊 Dataset Explorer & Quality Audit</h1>
        <p>Interactive table and quality metrics of the 91 smartphone records.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if df_smartphones is not None:
        st.subheader("Filter and Inspect Records")
        brand_filter = st.multiselect("Filter by Brand", options=df_smartphones["Phone_Company"].unique(), default=df_smartphones["Phone_Company"].unique()[:4])
        
        filtered = df_smartphones[df_smartphones["Phone_Company"].isin(brand_filter)] if brand_filter else df_smartphones
        st.dataframe(filtered, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Data Quality Audit Summary")
        qr_path = os.path.join(WORKSPACE_ROOT, "reports", "data_quality_report.csv")
        if os.path.exists(qr_path):
            qr_df = pd.read_csv(qr_path)
            st.dataframe(qr_df, use_container_width=True)


# ==========================================
# PAGE 6: EDA VISUALIZATIONS
# ==========================================
elif nav_page in ["EDA Visualization", "EDA Visualizations", "📈 EDA Visualizations"]:
    st.markdown("""
    <div class="main-header">
        <h1>📈 Exploratory Data Analysis Gallery</h1>
        <p>Full catalog of 16 publication-quality statistical distributions and relationship graphs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    fig_dir = os.path.join(WORKSPACE_ROOT, "reports", "figures")
    fig_files = sorted([f for f in os.listdir(fig_dir) if f.endswith(".png") and f[:2].isdigit()])
    
    selected_fig = st.selectbox("Select Visual Exploration Graph", fig_files)
    if selected_fig:
        st.image(os.path.join(fig_dir, selected_fig), use_container_width=True)
        
    st.markdown("---")
    st.subheader("Residual & Diagnostic Validation Charts")
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        p1 = os.path.join(fig_dir, "actual_vs_predicted.png")
        if os.path.exists(p1):
            st.image(p1, caption="Actual vs. Predicted Price Parity Plot", use_container_width=True)
    with d_col2:
        p2 = os.path.join(fig_dir, "residual_plot.png")
        if os.path.exists(p2):
            st.image(p2, caption="Residuals vs Predicted Price", use_container_width=True)

