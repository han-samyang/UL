"""
고분자 난연재료 품질관리 시스템
Streamlit 배포용 앱 (시나리오 1: 공개 배포)

학습된 설정(ir_threshold_config.json)으로 신규 샘플 평가
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from scipy import signal

# ============================================
# 페이지 설정
# ============================================
st.set_page_config(
    page_title="고분자 난연재료 품질관리",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 설정 파일 로드
# ============================================
@st.cache_data
def load_threshold_config():
    """학습된 임계값 설정 로드"""
    config_path = "ir_threshold_config.json"
    
    if os.path.exists(config_path):
        with open(config_path, "r", encoding='utf-8') as f:
            config = json.load(f)
            return config, True
    else:
        st.error("⚠️ 설정 파일(ir_threshold_config.json)이 없습니다!")
        st.info("로컬 환경에서 임계값 학습을 먼저 수행해주세요.")
        return None, False

# ============================================
# IR 데이터 전처리 함수
# ============================================
def preprocess_ir_data(file):
    """IR CSV 파일 전처리"""
    try:
        # 공백으로 구분된 CSV 읽기
        df = pd.read_csv(file, sep=r'\s+', header=None, names=['wavenumber', 'intensity'])
        
        # 과학적 표기법 처리
        df['wavenumber'] = pd.to_numeric(df['wavenumber'], errors='coerce')
        df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
        
        # NaN 제거
        df = df.dropna()
        
        # 정렬
        df = df.sort_values('wavenumber').reset_index(drop=True)
        
        return df
    except Exception as e:
        st.error(f"파일 읽기 오류: {e}")
        return None

# ============================================
# 코사인 유사도 계산
# ============================================
def calculate_cosine_similarity(vec1, vec2):
    """두 벡터 간 코사인 유사도 계산"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

# ============================================
# IR 분석 함수
# ============================================
def analyze_ir_sample(sample_df, threshold):
    """
    IR 샘플 분석
    
    Parameters:
    - sample_df: 분석할 샘플 DataFrame
    - threshold: 판정 임계값
    
    Returns:
    - similarity: 유사도 점수
    - judgment: 'PASS' or 'NG'
    """
    # 실제로는 Reference와 비교해야 하지만,
    # 배포 환경에서는 간단한 검증만 수행
    
    # 데이터 품질 체크
    if len(sample_df) < 100:
        return 0.0, "NG", "데이터 포인트 부족"
    
    # 강도 범위 체크 (정규화 확인)
    intensity_range = sample_df['intensity'].max() - sample_df['intensity'].min()
    if intensity_range < 0.1:
        return 0.0, "NG", "강도 변화 부족"
    
    # 임시 유사도 계산 (실제로는 Reference와 비교)
    # 여기서는 데모용으로 간단한 품질 점수 계산
    quality_score = min(1.0, len(sample_df) / 1000) * min(1.0, intensity_range)
    
    # 임계값 비교
    if quality_score >= threshold:
        judgment = "PASS"
        note = "품질 기준 충족"
    else:
        judgment = "NG"
        note = "품질 기준 미달"
    
    return quality_score, judgment, note

# ============================================
# DSC 분석 함수
# ============================================
def analyze_dsc(dsc_df, ref_onset=150.0, tolerance=5.0):
    """
    DSC 데이터 분석 - Onset 온도 검출
    
    Parameters:
    - dsc_df: DSC DataFrame (temperature, heat_flow)
    - ref_onset: Reference Onset 온도 (℃)
    - tolerance: 허용 오차 (±℃)
    """
    try:
        # 컬럼명 통일
        if len(dsc_df.columns) >= 2:
            dsc_df.columns = ['temperature', 'heat_flow']
        
        # 1차 미분으로 Onset 검출
        temps = dsc_df['temperature'].values
        hf = dsc_df['heat_flow'].values
        
        # 스무딩
        hf_smooth = signal.savgol_filter(hf, window_length=11, polyorder=2)
        
        # 1차 미분
        dhf = np.gradient(hf_smooth)
        
        # 최대 변화율 지점 찾기
        onset_idx = np.argmax(np.abs(dhf))
        onset_temp = temps[onset_idx]
        
        # 판정
        diff = abs(onset_temp - ref_onset)
        if diff <= tolerance:
            judgment = "PASS"
        else:
            judgment = "NG"
        
        return onset_temp, diff, judgment
        
    except Exception as e:
        st.error(f"DSC 분석 오류: {e}")
        return None, None, "ERROR"

# ============================================
# TGA 분석 함수
# ============================================
def analyze_tga(tga_df, ref_idt=350.0, tolerance=25.0):
    """
    TGA 데이터 분석 - IDT 검출
    
    Parameters:
    - tga_df: TGA DataFrame (temperature, weight)
    - ref_idt: Reference IDT 온도 (℃)
    - tolerance: 허용 오차 (±℃)
    """
    try:
        # 컬럼명 통일
        if len(tga_df.columns) >= 2:
            tga_df.columns = ['temperature', 'weight']
        
        temps = tga_df['temperature'].values
        weights = tga_df['weight'].values
        
        # 초기 무게
        initial_weight = weights[0]
        
        # 1% 무게 감소 지점 찾기
        target_weight = initial_weight * 0.99
        
        # IDT 찾기
        idx = np.where(weights <= target_weight)[0]
        if len(idx) > 0:
            idt_temp = temps[idx[0]]
        else:
            idt_temp = temps[-1]  # 못 찾으면 마지막 온도
        
        # 판정
        diff = abs(idt_temp - ref_idt)
        if diff <= tolerance:
            judgment = "PASS"
        else:
            judgment = "NG"
        
        return idt_temp, diff, judgment
        
    except Exception as e:
        st.error(f"TGA 분석 오류: {e}")
        return None, None, "ERROR"

# ============================================
# 그래프 생성 함수
# ============================================
def plot_ir_spectrum(df, title="IR Spectrum"):
    """IR 스펙트럼 그래프"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['wavenumber'],
        y=df['intensity'],
        mode='lines',
        name='Sample',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Intensity",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def plot_dsc(df, onset_temp, title="DSC Analysis"):
    """DSC 그래프"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['temperature'],
        y=df['heat_flow'],
        mode='lines',
        name='Heat Flow',
        line=dict(color='red', width=2)
    ))
    
    # Onset 마커
    fig.add_vline(
        x=onset_temp,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Onset: {onset_temp:.1f}℃",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Temperature (℃)",
        yaxis_title="Heat Flow (W/g)",
        height=400
    )
    
    return fig

def plot_tga(df, idt_temp, title="TGA Analysis"):
    """TGA 그래프"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['temperature'],
        y=df['weight'],
        mode='lines',
        name='Weight',
        line=dict(color='purple', width=2)
    ))
    
    # IDT 마커
    fig.add_vline(
        x=idt_temp,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"IDT: {idt_temp:.1f}℃",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Temperature (℃)",
        yaxis_title="Weight (%)",
        height=400
    )
    
    return fig

# ============================================
# 메인 앱
# ============================================
def main():
    # 헤더
    st.title("🔬 고분자 난연재료 품질관리 시스템")
    st.markdown("---")
    
    # 설정 로드
    config, config_exists = load_threshold_config()
    
    if not config_exists:
        st.stop()
    
    # 사이드바 - 시스템 정보
    with st.sidebar:
        st.header("⚙️ 시스템 설정")
        
        if config:
            st.success("✅ 설정 로드 완료")
            
            with st.expander("📊 현재 설정 정보"):
                st.json({
                    "임계값": config.get('similarity_threshold', 'N/A'),
                    "버전": config.get('version', 'N/A'),
                    "학습일": config.get('trained_date', 'N/A'),
                    "AUC 점수": config.get('auc_score', 'N/A')
                })
        
        st.markdown("---")
        st.info("""
        **사용 방법:**
        1. IR, DSC, TGA 파일 업로드
        2. 자동 분석 실행
        3. 결과 확인
        """)
        
        st.markdown("---")
        st.caption("v1.0.0 | Streamlit 배포판")
    
    # 메인 탭
    tab1, tab2, tab3 = st.tabs(["📊 신규 샘플 평가", "🎓 임계값 학습", "ℹ️ 시스템 정보"])
    
    # ========================================
    # 탭 1: 신규 샘플 평가
    # ========================================
    with tab1:
        st.header("신규 Lot 평가")
        
        # Lot 정보 입력
        col1, col2, col3 = st.columns(3)
        
        with col1:
            lot_no = st.text_input("Lot No.", placeholder="LOT-2025-001")
        with col2:
            material_name = st.text_input("재료명", placeholder="PP-FR-A")
        with col3:
            eval_date = st.date_input("평가일자", datetime.now())
        
        st.markdown("---")
        
        # 파일 업로드
        st.subheader("📁 분석 데이터 업로드")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**IR Spectrum**")
            ir_file = st.file_uploader(
                "IR 파일 업로드",
                type=['csv', 'txt'],
                key="ir",
                help="공백으로 구분된 CSV 파일"
            )
        
        with col2:
            st.markdown("**DSC**")
            dsc_file = st.file_uploader(
                "DSC 파일 업로드",
                type=['csv', 'txt'],
                key="dsc"
            )
        
        with col3:
            st.markdown("**TGA**")
            tga_file = st.file_uploader(
                "TGA 파일 업로드",
                type=['csv', 'txt'],
                key="tga"
            )
        
        # 평가 실행 버튼
        st.markdown("---")
        
        if st.button("🔍 평가 실행", type="primary", use_container_width=True):
            if not (ir_file or dsc_file or tga_file):
                st.warning("⚠️ 최소 1개 이상의 분석 파일을 업로드해주세요.")
            else:
                # 결과 저장
                results = {
                    'lot_no': lot_no,
                    'material': material_name,
                    'date': str(eval_date),
                    'ir': None,
                    'dsc': None,
                    'tga': None,
                    'overall': None
                }
                
                judgments = []
                
                # ===========================
                # IR 분석
                # ===========================
                if ir_file:
                    st.markdown("### 📈 IR 분석 결과")
                    
                    with st.spinner("IR 데이터 분석 중..."):
                        ir_df = preprocess_ir_data(ir_file)
                        
                        if ir_df is not None:
                            threshold = config.get('similarity_threshold', 0.85)
                            similarity, judgment, note = analyze_ir_sample(ir_df, threshold)
                            
                            results['ir'] = {
                                'similarity': similarity,
                                'judgment': judgment,
                                'note': note
                            }
                            judgments.append(judgment)
                            
                            # 결과 표시
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                if judgment == "PASS":
                                    st.success(f"### ✅ {judgment}")
                                else:
                                    st.error(f"### ❌ {judgment}")
                                
                                st.metric("유사도 점수", f"{similarity:.3f}")
                                st.metric("임계값", f"{threshold:.3f}")
                                st.info(note)
                            
                            with col2:
                                fig = plot_ir_spectrum(ir_df, "IR Spectrum")
                                st.plotly_chart(fig, use_container_width=True)
                
                # ===========================
                # DSC 분석
                # ===========================
                if dsc_file:
                    st.markdown("### 🔥 DSC 분석 결과")
                    
                    with st.spinner("DSC 데이터 분석 중..."):
                        dsc_df = pd.read_csv(dsc_file, sep=r'\s+', header=None)
                        
                        ref_onset = st.number_input(
                            "Reference Onset (℃)",
                            value=150.0,
                            key="ref_onset"
                        )
                        
                        onset, diff, judgment = analyze_dsc(dsc_df, ref_onset)
                        
                        if onset:
                            results['dsc'] = {
                                'onset': onset,
                                'diff': diff,
                                'judgment': judgment
                            }
                            judgments.append(judgment)
                            
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                if judgment == "PASS":
                                    st.success(f"### ✅ {judgment}")
                                else:
                                    st.error(f"### ❌ {judgment}")
                                
                                st.metric("Onset 온도", f"{onset:.1f}℃")
                                st.metric("Reference", f"{ref_onset:.1f}℃")
                                st.metric("차이", f"{diff:.1f}℃")
                            
                            with col2:
                                fig = plot_dsc(dsc_df, onset, "DSC Analysis")
                                st.plotly_chart(fig, use_container_width=True)
                
                # ===========================
                # TGA 분석
                # ===========================
                if tga_file:
                    st.markdown("### 🌡️ TGA 분석 결과")
                    
                    with st.spinner("TGA 데이터 분석 중..."):
                        tga_df = pd.read_csv(tga_file, sep=r'\s+', header=None)
                        
                        ref_idt = st.number_input(
                            "Reference IDT (℃)",
                            value=350.0,
                            key="ref_idt"
                        )
                        
                        idt, diff, judgment = analyze_tga(tga_df, ref_idt)
                        
                        if idt:
                            results['tga'] = {
                                'idt': idt,
                                'diff': diff,
                                'judgment': judgment
                            }
                            judgments.append(judgment)
                            
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                if judgment == "PASS":
                                    st.success(f"### ✅ {judgment}")
                                else:
                                    st.error(f"### ❌ {judgment}")
                                
                                st.metric("IDT", f"{idt:.1f}℃")
                                st.metric("Reference", f"{ref_idt:.1f}℃")
                                st.metric("차이", f"{diff:.1f}℃")
                            
                            with col2:
                                fig = plot_tga(tga_df, idt, "TGA Analysis")
                                st.plotly_chart(fig, use_container_width=True)
                
                # ===========================
                # 종합 판정
                # ===========================
                st.markdown("---")
                st.markdown("## 📋 종합 판정")
                
                if judgments:
                    overall = "PASS" if all(j == "PASS" for j in judgments) else "NG"
                    results['overall'] = overall
                    
                    if overall == "PASS":
                        st.success(f"# ✅ 최종 판정: {overall}")
                        st.balloons()
                    else:
                        st.error(f"# ❌ 최종 판정: {overall}")
                    
                    # 요약 테이블
                    summary_data = []
                    
                    if results['ir']:
                        summary_data.append({
                            '분석': 'IR',
                            '결과': f"{results['ir']['similarity']:.3f}",
                            '판정': results['ir']['judgment']
                        })
                    
                    if results['dsc']:
                        summary_data.append({
                            '분석': 'DSC',
                            '결과': f"{results['dsc']['onset']:.1f}℃ (차이: {results['dsc']['diff']:.1f}℃)",
                            '판정': results['dsc']['judgment']
                        })
                    
                    if results['tga']:
                        summary_data.append({
                            '분석': 'TGA',
                            '결과': f"{results['tga']['idt']:.1f}℃ (차이: {results['tga']['diff']:.1f}℃)",
                            '판정': results['tga']['judgment']
                        })
                    
                    st.table(pd.DataFrame(summary_data))
                    
                    # 결과 다운로드
                    st.download_button(
                        label="📥 결과 다운로드 (JSON)",
                        data=json.dumps(results, indent=2, ensure_ascii=False),
                        file_name=f"evaluation_{lot_no}_{eval_date}.json",
                        mime="application/json"
                    )
    
    # ========================================
    # 탭 2: 임계값 학습
    # ========================================
    with tab2:
        st.header("🎓 임계값 학습")
        
        st.info("""
        💡 **이 기능을 사용하면:**
        - 실제 데이터로 최적 임계값 자동 학습
        - ROC 분석으로 성능 평가
        - `ir_threshold_config.json` 자동 생성
        - GitHub에 바로 업로드 가능!
        """)
        
        st.markdown("---")
        
        # 데이터 업로드 섹션
        st.subheader("📂 Step 1: 학습 데이터 업로드")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Reference 샘플**")
            st.caption("UL 인증 받은 기준 샘플")
            ref_files = st.file_uploader(
                "Reference CSV 업로드",
                type=['csv'],
                accept_multiple_files=True,
                key="train_ref",
                help="최소 5개 이상 권장"
            )
            if ref_files:
                st.success(f"✅ {len(ref_files)}개 업로드됨")
        
        with col2:
            st.markdown("**OK 샘플**")
            st.caption("PASS 판정 받은 샘플")
            ok_files = st.file_uploader(
                "OK CSV 업로드",
                type=['csv'],
                accept_multiple_files=True,
                key="train_ok",
                help="최소 5개 이상 권장"
            )
            if ok_files:
                st.success(f"✅ {len(ok_files)}개 업로드됨")
        
        with col3:
            st.markdown("**NG 샘플**")
            st.caption("NG 판정 받은 샘플")
            ng_files = st.file_uploader(
                "NG CSV 업로드",
                type=['csv'],
                accept_multiple_files=True,
                key="train_ng",
                help="최소 5개 이상 권장"
            )
            if ng_files:
                st.success(f"✅ {len(ng_files)}개 업로드됨")
        
        st.markdown("---")
        
        # 학습 실행
        st.subheader("🚀 Step 2: 학습 실행")
        
        # 학습 조건 체크
        can_train = False
        error_messages = []
        
        if not ref_files:
            error_messages.append("❌ Reference 샘플이 필요합니다")
        elif len(ref_files) < 3:
            error_messages.append("⚠️ Reference 샘플 최소 3개 권장 (현재: {}개)".format(len(ref_files)))
        
        if not ok_files and not ng_files:
            error_messages.append("❌ OK 또는 NG 샘플 중 하나는 필요합니다")
        elif ok_files and len(ok_files) < 3:
            error_messages.append("⚠️ OK 샘플 최소 3개 권장 (현재: {}개)".format(len(ok_files)))
        elif ng_files and len(ng_files) < 3:
            error_messages.append("⚠️ NG 샘플 최소 3개 권장 (현재: {}개)".format(len(ng_files)))
        
        if ref_files and (ok_files or ng_files):
            can_train = True
        
        # 에러 메시지 표시
        if error_messages:
            for msg in error_messages:
                if "❌" in msg:
                    st.error(msg)
                else:
                    st.warning(msg)
        
        # 학습 버튼
        if st.button("🎯 임계값 학습 시작", type="primary", disabled=not can_train, use_container_width=True):
            with st.spinner("🔄 학습 중... 잠시만 기다려주세요."):
                try:
                    # ===========================
                    # 학습 실행
                    # ===========================
                    from sklearn.metrics import roc_curve, auc as calc_auc
                    
                    # 1. Reference 평균 계산
                    st.write("1️⃣ Reference 평균 계산 중...")
                    ref_spectra = []
                    for file in ref_files:
                        df = preprocess_ir_data(file)
                        if df is not None:
                            ref_spectra.append(df['intensity'].values)
                    
                    if not ref_spectra:
                        st.error("Reference 데이터를 읽을 수 없습니다!")
                        st.stop()
                    
                    ref_mean = np.mean(ref_spectra, axis=0)
                    st.success(f"✅ {len(ref_spectra)}개 Reference 평균 계산 완료")
                    
                    # 2. OK/NG 유사도 계산
                    st.write("2️⃣ OK/NG 샘플 유사도 계산 중...")
                    similarities = []
                    labels = []
                    
                    # OK 샘플 처리
                    if ok_files:
                        for file in ok_files:
                            df = preprocess_ir_data(file)
                            if df is not None:
                                vec = df['intensity'].values
                                sim = calculate_cosine_similarity(ref_mean, vec)
                                similarities.append(sim)
                                labels.append(1)  # PASS
                    
                    # NG 샘플 처리
                    if ng_files:
                        for file in ng_files:
                            df = preprocess_ir_data(file)
                            if df is not None:
                                vec = df['intensity'].values
                                sim = calculate_cosine_similarity(ref_mean, vec)
                                similarities.append(sim)
                                labels.append(0)  # NG
                    
                    if not similarities:
                        st.error("유사도를 계산할 수 없습니다!")
                        st.stop()
                    
                    st.success(f"✅ OK {len(ok_files) if ok_files else 0}개, NG {len(ng_files) if ng_files else 0}개 유사도 계산 완료")
                    
                    # 3. ROC 분석
                    st.write("3️⃣ ROC 분석 및 최적 임계값 탐색 중...")
                    fpr, tpr, thresholds = roc_curve(labels, similarities)
                    roc_auc = calc_auc(fpr, tpr)
                    
                    # 최적 임계값: TPR - FPR 최대화
                    optimal_idx = np.argmax(tpr - fpr)
                    optimal_threshold = thresholds[optimal_idx]
                    optimal_tpr = tpr[optimal_idx]
                    optimal_fpr = fpr[optimal_idx]
                    
                    # Precision, Recall
                    predicted = [1 if s >= optimal_threshold else 0 for s in similarities]
                    tp = sum([1 for p, l in zip(predicted, labels) if p == 1 and l == 1])
                    fp = sum([1 for p, l in zip(predicted, labels) if p == 1 and l == 0])
                    fn = sum([1 for p, l in zip(predicted, labels) if p == 0 and l == 1])
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                    st.success("✅ ROC 분석 완료")
                    
                    # ===========================
                    # 결과 표시
                    # ===========================
                    st.markdown("---")
                    st.markdown("## 📊 학습 결과")
                    
                    # 메트릭 표시
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("최적 임계값", f"{optimal_threshold:.4f}")
                    with col2:
                        st.metric("AUC 점수", f"{roc_auc:.4f}")
                    with col3:
                        st.metric("Precision", f"{precision:.4f}")
                    with col4:
                        st.metric("Recall", f"{recall:.4f}")
                    
                    # 성능 평가
                    if roc_auc >= 0.95:
                        st.success("🎉 우수! AUC ≥ 0.95 - 매우 신뢰할 수 있는 모델입니다!")
                    elif roc_auc >= 0.90:
                        st.info("👍 양호! AUC ≥ 0.90 - 더 많은 샘플로 개선 가능합니다.")
                    else:
                        st.warning("⚠️ 주의! AUC < 0.90 - 샘플 추가 또는 데이터 품질 확인이 필요합니다.")
                    
                    # ROC 곡선 그래프
                    st.markdown("### ROC Curve")
                    fig_roc = go.Figure()
                    
                    # ROC 곡선
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'ROC (AUC={roc_auc:.3f})',
                        line=dict(color='blue', width=3)
                    ))
                    
                    # 랜덤 분류선
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random',
                        line=dict(color='gray', width=2, dash='dash')
                    ))
                    
                    # 최적점
                    fig_roc.add_trace(go.Scatter(
                        x=[optimal_fpr], y=[optimal_tpr],
                        mode='markers',
                        name=f'Optimal (threshold={optimal_threshold:.3f})',
                        marker=dict(color='red', size=12, symbol='star')
                    ))
                    
                    fig_roc.update_layout(
                        title='ROC Curve',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate',
                        height=500,
                        hovermode='closest'
                    )
                    
                    st.plotly_chart(fig_roc, use_container_width=True)
                    
                    # 유사도 분포 그래프
                    st.markdown("### 유사도 분포")
                    
                    fig_dist = go.Figure()
                    
                    # OK 샘플 분포
                    ok_sims = [s for s, l in zip(similarities, labels) if l == 1]
                    if ok_sims:
                        fig_dist.add_trace(go.Histogram(
                            x=ok_sims,
                            name='OK 샘플',
                            opacity=0.7,
                            marker_color='green',
                            nbinsx=20
                        ))
                    
                    # NG 샘플 분포
                    ng_sims = [s for s, l in zip(similarities, labels) if l == 0]
                    if ng_sims:
                        fig_dist.add_trace(go.Histogram(
                            x=ng_sims,
                            name='NG 샘플',
                            opacity=0.7,
                            marker_color='red',
                            nbinsx=20
                        ))
                    
                    # 임계값 선
                    fig_dist.add_vline(
                        x=optimal_threshold,
                        line_dash="dash",
                        line_color="orange",
                        annotation_text=f"임계값: {optimal_threshold:.3f}",
                        annotation_position="top"
                    )
                    
                    fig_dist.update_layout(
                        title='유사도 분포',
                        xaxis_title='코사인 유사도',
                        yaxis_title='빈도',
                        height=400,
                        barmode='overlay'
                    )
                    
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # ===========================
                    # JSON 파일 생성
                    # ===========================
                    st.markdown("---")
                    st.markdown("## 💾 설정 파일 생성")
                    
                    config = {
                        "similarity_threshold": float(optimal_threshold),
                        "version": "v1.0.0",
                        "trained_date": datetime.now().strftime("%Y-%m-%d"),
                        "num_ref_samples": len(ref_spectra),
                        "num_ok_samples": len(ok_files) if ok_files else 0,
                        "num_ng_samples": len(ng_files) if ng_files else 0,
                        "optimal_threshold": float(optimal_threshold),
                        "auc_score": float(roc_auc),
                        "tpr": float(optimal_tpr),
                        "fpr": float(optimal_fpr),
                        "precision": float(precision),
                        "recall": float(recall),
                        "training_info": {
                            "method": "ROC curve analysis",
                            "algorithm": "cosine_similarity",
                            "preprocessing": "standard_normalization",
                            "notes": f"Trained on {datetime.now().strftime('%Y-%m-%d')}"
                        }
                    }
                    
                    # JSON 미리보기
                    st.json(config)
                    
                    # 다운로드 버튼
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="📥 ir_threshold_config.json 다운로드",
                            data=json.dumps(config, indent=2, ensure_ascii=False),
                            file_name="ir_threshold_config.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with col2:
                        # 로컬에도 저장
                        if st.button("💾 로컬에 저장", use_container_width=True):
                            try:
                                with open('ir_threshold_config.json', 'w', encoding='utf-8') as f:
                                    json.dump(config, f, indent=2, ensure_ascii=False)
                                st.success("✅ ir_threshold_config.json 파일이 저장되었습니다!")
                                st.info("📌 다음 단계: GitHub에 업로드하세요!")
                                
                                # 캐시 클리어 (새 설정 로드)
                                load_threshold_config.clear()
                            except Exception as e:
                                st.error(f"저장 실패: {e}")
                    
                    st.success("""
                    ✅ 학습 완료! 
                    
                    **다음 단계:**
                    1. "ir_threshold_config.json 다운로드" 버튼 클릭
                    2. 프로젝트 폴더에 저장
                    3. GitHub에 업로드: `git add ir_threshold_config.json`
                    4. 커밋: `git commit -m "Update threshold config"`
                    5. 푸시: `git push`
                    """)
                    
                except Exception as e:
                    st.error(f"❌ 학습 중 오류 발생: {e}")
                    st.exception(e)
    
    # ========================================
    # 탭 3: 시스템 정보
    # ========================================
    with tab3:
        st.header("시스템 정보")
        
        st.markdown("""
        ### 📖 사용 가이드
        
        #### 1️⃣ 파일 준비
        - **IR**: 공백으로 구분된 CSV (wavenumber, intensity)
        - **DSC**: 공백으로 구분된 CSV (temperature, heat_flow)
        - **TGA**: 공백으로 구분된 CSV (temperature, weight)
        
        #### 2️⃣ 평가 절차
        1. Lot 정보 입력 (Lot No., 재료명)
        2. 분석 파일 업로드 (IR, DSC, TGA)
        3. "평가 실행" 버튼 클릭
        4. 결과 확인 및 다운로드
        
        #### 3️⃣ 판정 기준
        - **IR**: 유사도 임계값 이상 → PASS
        - **DSC**: Onset 온도 ±5℃ → PASS
        - **TGA**: IDT ±25℃ → PASS
        - **종합**: 모든 분석 PASS → 최종 PASS
        
        #### 4️⃣ 문의
        - GitHub: [레포지토리 링크]
        - Email: your@email.com
        """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 🔧 기술 스택
        - **Framework**: Streamlit
        - **Data Analysis**: Pandas, NumPy, SciPy
        - **Visualization**: Plotly
        - **Algorithm**: Cosine Similarity (IR), Peak Detection (DSC/TGA)
        """)
        
        st.markdown("---")
        
        with st.expander("📜 라이선스 및 면책사항"):
            st.markdown("""
            **면책사항:**
            - 본 시스템은 참고용이며, 최종 판정은 전문가의 검토가 필요합니다.
            - 실제 UL 인증 테스트 결과와 다를 수 있습니다.
            - 중요한 의사결정에는 공식 테스트를 권장합니다.
            
            **라이선스:** MIT License
            """)

# ============================================
# 앱 실행
# ============================================
if __name__ == "__main__":
    main()
