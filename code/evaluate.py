"""
evaluate.py
-----------
학습된 멀티태스크 모델(best_model_*.pth)을 test set으로 전수 평가하고,
OpenCV Haar Cascade 기반 표정 검출과 정확도/정밀도/재현율을 비교한다.
평가 결과는 혼동행렬, 오류 분류 도넛차트 등 PPT용 그래프 6종으로 저장된다.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import torch

from config import CODE_DIR, PROCESSED_DIR, RESULTS_DIR
from dataset import get_dataloaders
from model import MultiTaskResNet

# 시각화 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def denormalize_image(tensor):
    image = tensor.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)

def draw_text_pil(img_rgb, c_true, c_pred, s_true, s_pred, cv_pred, class_names, emotion_names):
    """사진 위에 여백 없이 직접 3줄 텍스트 기입 (그림자 효과로 가독성 확보)"""
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("malgun.ttf", 14)
    except:
        font = ImageFont.load_default()

    c_mark = "O" if c_pred == c_true else "X"
    s_mark = "O" if s_pred == s_true else "X"
    cv_mark = "O" if cv_pred == s_true else "X"

    lines = [
        f"[AI] {class_names[c_true]} → {class_names[c_pred]} {c_mark}",
        f"[AI] {emotion_names[s_true]} → {emotion_names[s_pred]} {s_mark}",
        f"[CV] {emotion_names[cv_pred]} {cv_mark}"
    ]

    y_pos = 10
    for line in lines:
        for adj in [(-1,-1), (1,-1), (-1,1), (1,1)]:
            draw.text((10+adj[0], y_pos+adj[1]), line, font=font, fill=(0, 0, 0))
        draw.text((10, y_pos), line, font=font, fill=(255, 255, 255))
        y_pos += 22

    return np.array(pil_img)

def save_korean_image(file_path, img_bgr):
    ext = os.path.splitext(file_path)[1]
    result, nparr = cv2.imencode(ext, img_bgr)
    if result:
        with open(file_path, mode='wb') as f:
            nparr.tofile(f)

def plot_benchmark_charts(ai_metrics, cv_metrics, output_dir):
    labels = ['정확도(Accuracy)', '정밀도(Precision)', '재현율(Recall)']
    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width/2, ai_metrics, width, label='AI 모델(ResNet18)', color='royalblue', zorder=3)
    bars2 = ax.bar(x + width/2, cv_metrics, width, label='OpenCV(Haar)', color='lightgray', zorder=3)

    ax.set_ylabel('점수 (0.0 ~ 1.0)', fontsize=12, fontweight='bold')
    ax.set_title('AI 모델 vs OpenCV 알고리즘 성능 비교 (표정 분류)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.legend(loc='upper right', fontsize=10)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f'{h:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_ppt_benchmark_result.png'), bbox_inches='tight', dpi=300)
    plt.close()

def plot_celeb_accuracy(all_c_labels, all_c_preds, class_names, output_dir):
    accs = []
    for idx in range(len(class_names)):
        true_list = [p for p, t in zip(all_c_preds, all_c_labels) if t == idx]
        if len(true_list) == 0:
            accs.append(0)
            continue
        correct = sum(p == idx for p in true_list)
        accs.append(correct / len(true_list))

    colors = ['#2ecc71' if a >= 0.8 else '#e67e22' if a >= 0.5 else '#e74c3c' for a in accs]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(class_names, accs, color=colors, zorder=3)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel('정확도', fontsize=12, fontweight='bold')
    ax.set_title('연예인별 분류 정확도', fontsize=16, fontweight='bold')
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='80% 기준선')
    ax.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)
    ax.legend(fontsize=10)

    for bar, acc in zip(bars, accs):
        ax.annotate(f'{acc:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xticks(rotation=0, ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_ppt_celeb_accuracy.png'), bbox_inches='tight', dpi=300)
    plt.close()

def plot_error_analysis(all_c_labels, all_c_preds, all_s_labels, all_s_preds, class_names, output_dir):
    all_correct, wrong_name, wrong_smile, all_wrong = 0, 0, 0, 0
    for c_t, c_p, s_t, s_p in zip(all_c_labels, all_c_preds, all_s_labels, all_s_preds):
        if c_p == c_t and s_p == s_t: all_correct += 1
        elif c_p != c_t and s_p == s_t: wrong_name += 1
        elif c_p == c_t and s_p != s_t: wrong_smile += 1
        else: all_wrong += 1

    sizes = [all_correct, wrong_name, wrong_smile, all_wrong]
    percentages = [v / sum(sizes) * 100 for v in sizes] if sum(sizes) > 0 else [0, 0, 0, 0]
    colors_pie = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']

    # 3. 도넛 차트 (아이디어 1번: 좁은 구역은 지시선으로 바깥으로 빼기)
    fig1, ax1 = plt.subplots(figsize=(7, 7)) # 바깥쪽 텍스트가 잘리지 않게 도화지를 넓힘
    wedgeprops = {'width': 0.4, 'edgecolor': 'w', 'linewidth': 3}
    
    wedges, _ = ax1.pie(sizes, labels=None, autopct=None, startangle=90, wedgeprops=wedgeprops, colors=colors_pie)
    
    for i, w in enumerate(wedges):
        if sizes[i] == 0: continue
        
        pct = percentages[i]
        text_str = f'{pct:.1f}%'
        
        # 각 조각의 정중앙 각도 계산
        ang = (w.theta2 - w.theta1) / 2. + w.theta1
        rad = np.deg2rad(ang)
        x_dir, y_dir = np.cos(rad), np.sin(rad)
        
        if pct < 5.0: # 5% 미만인 좁은 조각은 바깥으로 뺌
            # 도넛 바깥쪽 테두리 좌표
            x_edge = w.r * x_dir
            y_edge = w.r * y_dir
            
            # 텍스트가 위치할 도넛 바깥 좌표 (1.25배 밖으로)
            x_ext = 1.25 * x_dir
            y_ext = 1.25 * y_dir
            
            # 좌우 위치에 따라 텍스트 정렬 방향 결정
            ha = 'left' if x_dir > 0 else 'right'
            
            # 얇은 지시선(arrowprops)을 사용하여 바깥에 텍스트 그리기
            ax1.annotate(text_str, xy=(x_edge, y_edge), xytext=(x_ext, y_ext),
                         ha=ha, va='center', fontsize=14, fontweight='bold', color=colors_pie[i],
                         arrowprops=dict(arrowstyle="-", color="gray", lw=1.2))
        else: # 넓은 조각은 도넛 안쪽 중앙에 배치
            r_in, r_out = w.r - w.width, w.r
            mean_r = (r_in + r_out) / 2
            x_center = mean_r * x_dir
            y_center = mean_r * y_dir
            ax1.text(x_center, y_center, text_str,
                     ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    ax1.set_title('4단계 오류 분류 비율', fontsize=16, fontweight='bold', pad=15)

    # 하단 범례 추가
    legend_labels = [
        f'전체 정답 ({sizes[0]}장)',
        f'이름 오답 ({sizes[1]}장)',
        f'표정 오답 ({sizes[2]}장)',
        f'전체 오답 ({sizes[3]}장)'
    ]
    ax1.legend(wedges, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), 
               ncol=2, fontsize=12, frameon=False)

    plt.savefig(os.path.join(output_dir, '03_ppt_error_donut.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # 4. 오답 원인 TOP 5 가로 막대 그래프
    mistake_map = {}
    for c_t, c_p in zip(all_c_labels, all_c_preds):
        if c_t != c_p:
            key = f"{class_names[c_t]} → {class_names[c_p]}"
            mistake_map[key] = mistake_map.get(key, 0) + 1

    sorted_mistakes = sorted(mistake_map.items(), key=lambda x: x[1], reverse=True)[:5]
    mistake_labels = [item[0] for item in sorted_mistakes]
    mistake_counts = [item[1] for item in sorted_mistakes]

    mistake_labels.reverse()
    mistake_counts.reverse()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    if mistake_counts:
        colors = sns.color_palette("Reds", len(mistake_counts))
        bars = ax2.barh(mistake_labels, mistake_counts, height=0.5, color=colors, alpha=0.9, zorder=3)
        
        ax2.set_title('가장 많이 헷갈린 연예인 (TOP 5)', fontsize=16, fontweight='bold', pad=15)
        ax2.set_xlabel('오분류 횟수 (장)', fontsize=13, fontweight='bold')
        
        max_val = max(mistake_counts)
        ax2.set_xlim(0, max_val * 1.2)
        ax2.xaxis.get_major_locator().set_params(integer=True) 
        
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)

        ax2.tick_params(axis='y', labelsize=12)
        ax2.tick_params(axis='x', labelsize=11)
        for label in ax2.get_yticklabels() + ax2.get_xticklabels():
            label.set_fontweight('bold')

        for bar, count in zip(bars, mistake_counts):
            ax2.text(bar.get_width() + (max_val * 0.01), bar.get_y() + bar.get_height()/2,
                     f'{count}장', va='center', fontsize=12, fontweight='bold', color='#333333')
    else:
        ax2.text(0.5, 0.5, "완벽합니다! 이름 오분류가 없습니다.", ha='center', va='center', fontsize=14)
        ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_ppt_error_top5.png'), bbox_inches='tight', dpi=300)
    plt.close()

def plot_confusion_matrices(all_c_labels, all_c_preds, all_s_labels, all_s_preds, class_names, emotion_names, output_dir):
    # 5. 연예인 분류 혼동 행렬
    plt.figure(figsize=(14, 8)) 
    
    cm_celeb = confusion_matrix(all_c_labels, all_c_preds)
    
    sns.heatmap(cm_celeb, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, 
                cbar=True, linewidths=2.0, linecolor='white', 
                annot_kws={"size": 15, "weight": "bold"})

    plt.title('연예인 분류 혼동 행렬', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('실제 정답', fontsize=14, fontweight='bold')
    plt.xlabel('AI 예측', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=0, ha='center', fontsize=10, fontweight='bold') 
    plt.yticks(rotation=0, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_ppt_confusion_matrix_celeb.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # 6. 표정 분류 혼동 행렬
    plt.figure(figsize=(6, 5)) 
    
    cm_smile = confusion_matrix(all_s_labels, all_s_preds)
    sns.heatmap(cm_smile, annot=True, fmt='d', cmap='Greens',
                xticklabels=emotion_names, yticklabels=emotion_names, 
                cbar=True, linewidths=2.0, linecolor='white',
                annot_kws={"size": 16, "weight": "bold"})

    plt.title('표정 분류 혼동 행렬', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('실제 정답', fontsize=12, fontweight='bold')
    plt.xlabel('AI 예측', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=0, ha='center', fontsize=12, fontweight='bold')
    plt.yticks(rotation=0, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_ppt_confusion_matrix_smile.png'), bbox_inches='tight', dpi=300)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="학습된 모델 전수 평가 및 OpenCV 벤치마크")
    parser.add_argument("--data_dir", type=str, default=PROCESSED_DIR, help="얼굴 크롭까지 끝난 train/val/test 데이터 경로 (preprocessing.py 결과물)")
    parser.add_argument("--model_path", type=str, default=os.path.join(CODE_DIR, "best_model_a1.0_b0.5.pth"))
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR)
    return parser.parse_args()


def evaluate_and_benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = args.data_dir
    output_dir = args.output_dir
    model_path = args.model_path

    for folder in ['All_Correct', 'Wrong_Name', 'Wrong_Smile', 'All_Wrong']:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    print("▶ 4단계 폴더 생성 완료")

    dataloaders, class_names = get_dataloaders(data_dir, batch_size=1)
    test_loader = dataloaders.get('test', dataloaders.get('val'))
    emotion_names = ['Neutral', 'Smile']

    model = MultiTaskResNet(num_celebs=len(class_names)).to(device)

    if not os.path.exists(model_path):
        print(f"❌ {model_path} 파일이 없습니다.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_c_labels, all_c_preds = [], []
    all_s_labels, all_s_preds = [], []
    opencv_s_preds = []

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    print(f"▶ 전수 평가 시작...")

    with torch.no_grad():
        for i, (inputs, c_labels, s_labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            c_true, s_true = c_labels.item(), s_labels.item()

            c_out, s_out = model(inputs)
            c_pred = torch.argmax(c_out, 1).item()
            s_pred = torch.argmax(s_out, 1).item()

            all_c_labels.append(c_true)
            all_c_preds.append(c_pred)
            all_s_labels.append(s_true)
            all_s_preds.append(s_pred)

            img_rgb = denormalize_image(inputs[0])
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            cv_pred = 0
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                if len(smiles) > 0:
                    cv_pred = 1
                    break
            opencv_s_preds.append(cv_pred)

            result_img = draw_text_pil(
                img_rgb, c_true, c_pred, s_true, s_pred, cv_pred,
                class_names, emotion_names
            )

            if c_pred == c_true and s_pred == s_true: folder = 'All_Correct'
            elif c_pred != c_true and s_pred == s_true: folder = 'Wrong_Name'
            elif c_pred == c_true and s_pred != s_true: folder = 'Wrong_Smile'
            else: folder = 'All_Wrong'

            save_fn = f"{i+1}_{class_names[c_true]}.jpg"
            save_path = os.path.join(output_dir, folder, save_fn)
            save_korean_image(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

    # 지표 산출
    ai_c_acc = accuracy_score(all_c_labels, all_c_preds)
    ai_s_acc = accuracy_score(all_s_labels, all_s_preds)
    ai_s_prec = precision_score(all_s_labels, all_s_preds, zero_division=0)
    ai_s_rec = recall_score(all_s_labels, all_s_preds, zero_division=0)

    cv_s_acc = accuracy_score(all_s_labels, opencv_s_preds)
    cv_s_prec = precision_score(all_s_labels, opencv_s_preds, zero_division=0)
    cv_s_rec = recall_score(all_s_labels, opencv_s_preds, zero_division=0)

    print("\n" + "="*60)
    print(f"[AI] 이름 정확도: {ai_c_acc:.4f} | 표정 정확도: {ai_s_acc:.4f}")
    print(f"[CV] 표정 정확도: {cv_s_acc:.4f}")
    print("="*60)

    # 차트 개별 저장 실행
    plot_benchmark_charts([ai_s_acc, ai_s_prec, ai_s_rec], [cv_s_acc, cv_s_prec, cv_s_rec], output_dir)
    plot_celeb_accuracy(all_c_labels, all_c_preds, class_names, output_dir)
    plot_error_analysis(all_c_labels, all_c_preds, all_s_labels, all_s_preds, class_names, output_dir)
    plot_confusion_matrices(all_c_labels, all_c_preds, all_s_labels, all_s_preds, class_names, emotion_names, output_dir)

    print(f"\n✅ 완료! '{output_dir}' 폴더 안에 총 6장의 개별 그래프가 고화질로 저장되었습니다.")

if __name__ == '__main__':
    evaluate_and_benchmark(parse_args())