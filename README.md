# 🔬 고분자 난연재료 품질관리 시스템

UL 인증 고분자 난연재료의 첨가제 변경 시 IR, DSC, TGA 분석 데이터를 기반으로 Pass/NG를 사전 평가하는 시스템입니다.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 📋 주요 기능

- ✅ **IR 스펙트럼 분석**: 코사인 유사도 기반 자동 판정
- ✅ **DSC 분석**: Onset 온도 자동 검출 및 비교 (±5℃)
- ✅ **TGA 분석**: IDT 자동 검출 및 비교 (±25℃)
- ✅ **임계값 학습**: 실제 데이터로 최적 임계값 자동 학습 🆕
- ✅ **ROC 분석**: 성능 평가 및 시각화
- ✅ **종합 판정**: 3가지 분석 결과 통합 평가
- ✅ **실시간 시각화**: Plotly 기반 인터랙티브 그래프
- ✅ **결과 다운로드**: JSON 형식 평가 리포트

## 🚀 빠른 시작

### 온라인 버전 (권장)

👉 [Streamlit 앱 바로가기](https://your-app-url.streamlit.app)

파일만 업로드하면 즉시 평가 가능!

### 로컬 실행

```bash
# 1. 레포지토리 클론
git clone https://github.com/your-username/polymer-qc-system.git
cd polymer-qc-system

# 2. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. Streamlit 실행
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

## 📁 프로젝트 구조

```
polymer-qc-system/
├── app.py                        # Streamlit 메인 앱
├── ir_threshold_config.json      # 학습된 임계값 설정
├── requirements.txt              # Python 패키지 목록
├── .gitignore                    # Git 제외 파일
├── README.md                     # 이 파일
└── data/                         # 데이터 폴더 (로컬만)
    ├── IR/
    │   ├── ref/                  # Reference 샘플
    │   ├── ok/                   # PASS 샘플
    │   └── ng/                   # NG 샘플
    ├── DSC/
    └── TGA/
```

## 📊 사용 방법

### 방법 1: 임계값 학습 (앱 내에서) 🆕

**Streamlit 앱에서 직접 학습:**

1. **앱 실행**
   ```bash
   streamlit run app.py
   ```

2. **"🎓 임계값 학습" 탭 이동**

3. **학습 데이터 업로드**
   - Reference 샘플: 최소 3-5개 (권장: 10-30개)
   - OK 샘플: 최소 3-5개 (권장: 10-20개)
   - NG 샘플: 최소 3-5개 (권장: 10-15개)

4. **"임계값 학습 시작" 버튼 클릭**
   - 자동으로 ROC 분석 수행
   - 최적 임계값 계산
   - 그래프로 결과 시각화

5. **설정 파일 다운로드**
   - `ir_threshold_config.json` 다운로드
   - 프로젝트 폴더에 저장

6. **GitHub 업데이트**
   ```bash
   git add ir_threshold_config.json
   git commit -m "Update threshold config"
   git push
   ```

---

### 방법 2: 신규 샘플 평가

### 1️⃣ 파일 준비

**IR 데이터 (CSV)**
```
4000.0 0.95
3999.5 0.94
3999.0 0.93
...
```
- 형식: 공백 구분
- 열1: Wavenumber (cm⁻¹)
- 열2: Intensity

**DSC 데이터 (CSV)**
```
25.0 -0.5
30.0 -0.6
35.0 -0.8
...
```
- 형식: 공백 구분
- 열1: Temperature (℃)
- 열2: Heat Flow (W/g)

**TGA 데이터 (CSV)**
```
25.0 100.0
50.0 99.8
75.0 99.5
...
```
- 형식: 공백 구분
- 열1: Temperature (℃)
- 열2: Weight (%)

### 2️⃣ 평가 실행

1. **Lot 정보 입력**
   - Lot No. (예: LOT-2025-001)
   - 재료명 (예: PP-FR-A)
   - 평가일자

2. **파일 업로드**
   - IR, DSC, TGA 중 1개 이상

3. **평가 실행 버튼 클릭**
   - 자동 분석 시작
   - 실시간 그래프 표시

4. **결과 확인**
   - ✅ PASS / ❌ NG 판정
   - 상세 수치 및 그래프
   - JSON 결과 다운로드

### 3️⃣ 판정 기준

| 분석 | 기준 | 허용 범위 |
|------|------|-----------|
| IR   | 유사도 | ≥ 0.87 (ROC 최적값) |
| DSC  | Onset 온도 | ±5℃ |
| TGA  | IDT | ±25℃ |
| 종합 | 모든 분석 | 전체 PASS 시 합격 |

## 🔧 기술 스택

- **Framework**: Streamlit 1.30+
- **Data Processing**: Pandas, NumPy
- **Scientific Computing**: SciPy
- **Visualization**: Plotly
- **Algorithm**: 
  - IR: Cosine Similarity
  - DSC/TGA: Peak Detection

## 📝 개발자 가이드

### 로컬 환경에서 임계값 재학습

```python
# threshold_trainer.py (별도 스크립트)
import json
import numpy as np
from sklearn.metrics import roc_curve, auc

# 데이터 로드
ref_data = load_reference_data()
ok_data = load_ok_data()
ng_data = load_ng_data()

# 유사도 계산
# ... (기존 학습 코드)

# 최적 임계값 찾기
fpr, tpr, thresholds = roc_curve(labels, similarities)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# 설정 저장
config = {
    "similarity_threshold": float(optimal_threshold),
    "version": "v1.1.0",
    "trained_date": datetime.now().strftime("%Y-%m-%d"),
    "auc_score": float(auc(fpr, tpr)),
    # ...
}

with open("ir_threshold_config.json", "w") as f:
    json.dump(config, f, indent=2)
```

### GitHub에 배포

```bash
# 1. 변경사항 커밋
git add .
git commit -m "Update threshold config to v1.1.0"

# 2. GitHub에 푸시
git push origin main

# 3. Streamlit Cloud에서 자동 재배포
# (연결되어 있으면 자동, 없으면 수동 재배포)
```

## ⚠️ 주의사항

### 배포 시 데이터 관리

**✅ GitHub에 포함되는 것:**
- `ir_threshold_config.json` (학습된 설정)
- 소스 코드 (`app.py` 등)
- 설정 파일 (`requirements.txt`)

**❌ GitHub에 제외되는 것:**
- `data/` 폴더의 실제 데이터 (회사 기밀)
- 사용자 업로드 파일
- 임시 파일

`.gitignore`에 의해 자동 제외됨!

### 보안

- 민감한 데이터는 절대 커밋하지 마세요
- `.gitignore`를 반드시 확인하세요
- Streamlit Secrets에 API 키 등 저장

## 🤝 기여

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

MIT License - 자유롭게 사용 가능

## 📧 연락처

- GitHub: [@your-username](https://github.com/your-username)
- Email: your.email@example.com

## 🙏 감사의 말

- UL 인증 기준 참고: [UL Standard](https://www.ul.com)
- Streamlit: [streamlit.io](https://streamlit.io)

---

**면책사항**: 본 시스템은 참고용이며, 최종 판정은 전문가 검토 및 공식 UL 테스트를 권장합니다.
