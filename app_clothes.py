import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
import io
import numpy as np
import colorsys
from collections import Counter
import base64
import os
from openai import OpenAI
from dotenv import load_dotenv

# 配置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 加载环境变量
load_dotenv()

# DeepSeek API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

_client = None

def get_client():
    global _client
    if _client is None and DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your_deepseek_api_key_here":
        _client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
    return _client

# 解决Windows中文乱码
def get_font(size):
    try:
        return ImageFont.truetype("msyh.ttc", size)
    except:
        return ImageFont.load_default(size=size)

# HSV转十六进制色值
def hsv_to_hex(h_deg, s=0.6, v=0.8):
    h = h_deg / 360
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

# ===================== 匹配你提供专业色卡：标准色彩数据库 =====================
# 格式：(颜色中文名称, 标准色相h, 标准饱和度s, 标准明度v)
STANDARD_COLOR_DB = [
    # 无黑白灰基底系（对应色卡左上角20%灰~黑色、米白）
    ("纯白", 0, 0, 0.98),
    ("20%灰", 0, 0, 0.8),
    ("40%灰", 0, 0, 0.6),
    ("60%灰", 0, 0, 0.4),
    ("80%灰", 0, 0, 0.2),
    ("黑色", 0, 0, 0.08),
    ("米白", 40, 0.08, 0.96),
    # 大地百搭色系：驼、卡其、燕麦、姜黄、棕、焦糖
    ("燕麦色", 35, 0.22, 0.92),
    ("米杏色", 38, 0.25, 0.90),
    ("浅驼", 42, 0.30, 0.86),
    ("驼色", 45, 0.38, 0.80),
    ("卡其", 48, 0.42, 0.76),
    ("姜黄", 52, 0.55, 0.72),
    ("土黄", 55, 0.60, 0.68),
    ("焦糖棕", 30, 0.65, 0.55),
    ("深棕", 25, 0.70, 0.40),
    # 红色系
    ("浅粉", 350, 0.20, 0.92),
    ("裸粉", 348, 0.30, 0.88),
    ("豆沙红", 345, 0.45, 0.75),
    ("砖红", 10, 0.60, 0.68),
    ("正红", 0, 0.75, 0.65),
    ("酒红", 355, 0.80, 0.45),
    # 橙色系
    ("蜜桃橙", 20, 0.35, 0.90),
    ("浅橙", 22, 0.45, 0.85),
    ("暖橙", 25, 0.60, 0.78),
    ("橘棕", 28, 0.68, 0.62),
    # 黄色/绿色系
    ("奶黄", 60, 0.18, 0.95),
    ("浅黄", 58, 0.30, 0.90),
    ("鹅黄", 55, 0.40, 0.86),
    ("浅绿", 110, 0.25, 0.88),
    ("薄荷绿", 120, 0.35, 0.82),
    ("草绿", 115, 0.50, 0.75),
    ("墨绿", 130, 0.65, 0.45),
    # 蓝/紫色系
    ("婴儿蓝", 210, 0.20, 0.90),
    ("雾霾蓝", 215, 0.35, 0.78),
    ("天青蓝", 220, 0.45, 0.72),
    ("藏蓝", 230, 0.70, 0.35),
    ("浅紫", 260, 0.25, 0.88),
    ("香芋紫", 265, 0.40, 0.76),
    ("葡萄紫", 270, 0.60, 0.58),
    ("深紫", 275, 0.72, 0.42),
]
# 拆分全部色名用于下拉选择
ALL_STD_COLOR_NAMES = [item[0] for item in STANDARD_COLOR_DB]
# 色相快速映射字典
COLOR_HSV_MAP = {name: (h, s, v) for name, h, s, v in STANDARD_COLOR_DB}
BASE_GRAY_COLORS = ["纯白", "20%灰", "40%灰", "60%灰", "80%灰", "黑色", "米白"]
EARTH_COLORS = ["燕麦色", "米杏色", "浅驼", "驼色", "卡其", "姜黄", "土黄", "焦糖棕", "深棕"]
WARM_COLORS = ["浅粉", "裸粉", "豆沙红", "砖红", "正红", "酒红", "蜜桃橙", "浅橙", "暖橙", "橘棕", "奶黄", "浅黄", "鹅黄"]
COLD_COLORS = ["浅绿", "薄荷绿", "草绿", "墨绿", "婴儿蓝", "雾霾蓝", "天青蓝", "藏蓝", "浅紫", "香芋紫", "葡萄紫", "深紫"]

# ===================== 使用DeepSeek模型进行颜色识别 =====================
def get_img_main_color_name_with_deepseek(pil_img):
    """
    使用DeepSeek模型优化颜色识别结果
    通过分析图片特征描述，使用AI进行智能颜色判断
    """
    try:
        # 先使用机械式识别获取颜色特征（过滤背景影响）
        small = pil_img.resize((140, 140))
        img_array = np.array(small)
        
        # 提取有效颜色特征（过滤背景）
        valid_pixels = []
        for y in range(img_array.shape[0]):
            for x in range(img_array.shape[1]):
                r, g, b = img_array[y, x]
                # 过滤纯白色背景和纯黑色
                if (r > 248 and g > 248 and b > 248) or (r < 8 and g < 8 and b < 8):
                    continue
                valid_pixels.append((r, g, b))
        
        # 如果过滤后没有有效像素，使用原图平均
        if len(valid_pixels) > 0:
            r_avg = np.mean([p[0] for p in valid_pixels]) / 255.0
            g_avg = np.mean([p[1] for p in valid_pixels]) / 255.0
            b_avg = np.mean([p[2] for p in valid_pixels]) / 255.0
        else:
            r_avg = np.mean(img_array[:, :, 0]) / 255.0
            g_avg = np.mean(img_array[:, :, 1]) / 255.0
            b_avg = np.mean(img_array[:, :, 2]) / 255.0
        
        # RGB转HSV
        h, s, v = colorsys.rgb_to_hsv(r_avg, g_avg, b_avg)
        h_deg = int(h * 360)
        s_percent = int(s * 100)
        v_percent = int(v * 100)
        
        # 判断颜色倾向描述
        color_tendency = ""
        if h_deg >= 0 and h_deg < 30:
            color_tendency = "红色倾向"
        elif h_deg >= 30 and h_deg < 60:
            color_tendency = "橙色倾向"
        elif h_deg >= 60 and h_deg < 90:
            color_tendency = "黄色倾向"
        elif h_deg >= 90 and h_deg < 150:
            color_tendency = "绿色倾向"
        elif h_deg >= 150 and h_deg < 180:
            color_tendency = "青色倾向"
        elif h_deg >= 180 and h_deg < 240:
            color_tendency = "蓝色倾向"
        elif h_deg >= 240 and h_deg < 300:
            color_tendency = "紫色倾向"
        elif h_deg >= 300 and h_deg < 360:
            color_tendency = "品红色倾向"
        
        # 明度描述
        brightness = ""
        if v_percent > 85:
            brightness = "高明度（浅色）"
        elif v_percent > 55:
            brightness = "中明度（适中）"
        elif v_percent > 30:
            brightness = "低明度（深色）"
        else:
            brightness = "极低明度（近黑色）"
        
        # 饱和度描述
        saturation = ""
        if s_percent < 15:
            saturation = "低饱和度（灰色调）"
        elif s_percent < 40:
            saturation = "中低饱和度"
        elif s_percent < 70:
            saturation = "中饱和度"
        else:
            saturation = "高饱和度（鲜艳）"
        
        # 构建prompt，使用纯文本描述颜色特征
        prompt = f"""你是一位专业的服装色彩分析师。根据以下颜色特征分析，请从专业色卡中选择最匹配的衣物颜色名称。

颜色特征分析：
- 色相角度：{h_deg}°（{color_tendency}）
- 饱和度：{s_percent}%（{saturation}）
- 明度：{v_percent}%（{brightness}）
- RGB值：R={int(r_avg*255)}, G={int(g_avg*255)}, B={int(b_avg*255)}

专业色卡颜色列表：
- 无黑白灰基底系：纯白、20%灰、40%灰、60%灰、80%灰、黑色、米白
- 大地百搭色系：燕麦色、米杏色、浅驼、驼色、卡其、姜黄、土黄、焦糖棕、深棕
- 红色系：浅粉、裸粉、豆沙红、砖红、正红、酒红
- 橙色系：蜜桃橙、浅橙、暖橙、橘棕
- 黄色/绿色系：奶黄、浅黄、鹅黄、浅绿、薄荷绿、草绿、墨绿
- 蓝/紫色系：婴儿蓝、雾霾蓝、天青蓝、藏蓝、浅紫、香芋紫、葡萄紫、深紫

请根据这些特征，从上述色卡中选择最匹配的颜色名称。
请直接返回颜色名称，不要有任何其他解释或说明。例如：驼色"""

        # 调用DeepSeek API（纯文本方式）
        response = get_client().chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        # 获取识别结果
        color_name = response.choices[0].message.content.strip()
        
        # 验证颜色名称是否在标准色卡中
        if color_name in ALL_STD_COLOR_NAMES:
            return color_name
        else:
            return "驼色"
            
    except Exception as e:
        # 记录详细错误信息但不显示给用户
        st.warning(f"DeepSeek API调用失败，将使用备用算法: {str(e)[:100]}...")
        raise e  # 重新抛出异常，让上层处理

# ===================== 保留原有的机械式识别函数作为备用 =====================
def get_img_main_color_name(pil_img, sample_step=6):
    """
    备用函数：使用机械式HSV颜色匹配算法识别颜色
    当DeepSeek API不可用时使用
    """
    small = pil_img.resize((140, 140))
    img_array = np.array(small)
    pixel_hsv_data = []
    gray_pixel_count = 0
    total_pixel = 0
    # 逐像素采样，过滤纯白背景、纯黑水印
    for y in range(0, img_array.shape[0], sample_step):
        for x in range(0, img_array.shape[1], sample_step):
            r, g, b = img_array[y, x] / 255.0
            total_pixel += 1
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            h_deg = h * 360
            # 过滤纯白背景、纯黑文字水印、极低明度脏点
            if v > 0.97 or v < 0.06:
                continue
            # 判断无彩色灰度像素
            if s < 0.10:
                gray_pixel_count += 1
                pixel_hsv_data.append(("gray", h_deg, s, v))
            else:
                pixel_hsv_data.append(("color", h_deg, s, v))
    # 场景1：图片主体为黑白灰无彩色衣物
    gray_ratio = gray_pixel_count / max(total_pixel, 1)
    if gray_ratio > 0.62:
        avg_v = np.mean([v for t, h, s, v in pixel_hsv_data])
        if avg_v > 0.92:
            return "纯白"
        elif avg_v > 0.75:
            return "20%灰"
        elif avg_v > 0.55:
            return "40%灰"
        elif avg_v > 0.35:
            return "60%灰"
        elif avg_v > 0.15:
            return "80%灰"
        else:
            return "黑色"
    # 场景2：彩色衣物，匹配标准色库最接近的颜色
    color_only_hsv = [(h, s, v) for t, h, s, v in pixel_hsv_data if t == "color"]
    if len(color_only_hsv) == 0:
        return "米白"
    # 统计出现频次最高的主色相
    h_list = [int(h) for h, s, v in color_only_hsv]
    h_counter = Counter(h_list)
    main_h = h_counter.most_common(1)[0][0]
    # 遍历标准色库，计算色相+饱和度+明度综合差值，取最匹配色
    min_diff = 9999
    match_color_name = "驼色"
    for std_name, std_h, std_s, std_v in STANDARD_COLOR_DB:
        if std_name in BASE_GRAY_COLORS:
            continue
        # 色相环形差值
        h_diff = abs((main_h - std_h + 180) % 360 - 180)
        # 饱和度、明度差值加权
        s_avg = np.mean([s for h, s, v in color_only_hsv])
        v_avg = np.mean([v for h, s, v in color_only_hsv])
        s_diff = abs(s_avg - std_s) * 120
        v_diff = abs(v_avg - std_v) * 90
        total_diff = h_diff + s_diff + v_diff
        if total_diff < min_diff:
            min_diff = total_diff
            match_color_name = std_name
    return match_color_name

# ===================== 智能颜色识别函数（优先使用DeepSeek，备用机械式） =====================
def smart_color_recognition(pil_img):
    """
    智能颜色识别：
    1. 优先使用DeepSeek模型进行识别
    2. 如果DeepSeek API不可用或失败，使用备用的机械式识别
    """
    # 检查DeepSeek API是否配置
    if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your_deepseek_api_key_here":
        try:
            return get_img_main_color_name_with_deepseek(pil_img)
        except Exception as e:
            st.warning(f"DeepSeek识别失败，使用备用算法: {str(e)}")
            return get_img_main_color_name(pil_img)
    else:
        st.info("未配置DeepSeek API，使用备用机械式识别算法")
        return get_img_main_color_name(pil_img)

# 6套专业配色逻辑
def get_same_tone_match(main_color_name):
    h, s, v = COLOR_HSV_MAP[main_color_name]
    group = [(h, s+0.12, v+0.08), (h, s-0.08, v-0.06), (h, s-0.15, v+0.10)]
    desc = "同色系深浅搭配｜同色相渐变，柔和不出错，通勤日常首选"
    explain = "上衣下装使用同一色系深浅，视觉统一干净，适配约会、校园穿搭"
    return group, desc, explain

def get_adjacent_match(main_color_name):
    main_h, _, _ = COLOR_HSV_MAP[main_color_name]
    left_h = (main_h - 40) % 360
    right_h = (main_h + 40) % 360
    group = [("邻近浅色调", left_h, 0.45, 0.85), ("邻近深色调", right_h, 0.50, 0.70)]
    desc = "邻近柔和撞色｜色相相邻，有层次不杂乱，休闲活力穿搭"
    explain = "上下装色彩接近但有区分，韩系、少年感穿搭专用"
    return group, desc, explain

def get_complement_match(main_color_name):
    main_h, _, _ = COLOR_HSV_MAP[main_color_name]
    comp_h = (main_h + 180) % 360
    group = [("互补撞色", comp_h, 0.55, 0.72)]
    desc = "互补对比配色｜180°对冲色彩，吸睛出片，度假拍照专用"
    explain = "仅小面积使用互补色做点缀，大面积容易俗气，适合旅行穿搭"
    return group, desc, explain

def get_triangle_match(main_color_name):
    main_h, _, _ = COLOR_HSV_MAP[main_color_name]
    h1 = (main_h + 120) % 360
    h2 = (main_h + 240) % 360
    group = [("三角色1", h1, 0.48, 0.78), ("三角色2", h2, 0.52, 0.66)]
    desc = "三色均衡三角搭配｜复古元气，潮流街头穿搭"
    explain = "三色均匀分配面积，适合复古、美式休闲穿搭，层次丰富"
    return group, desc, explain

def get_split_complement(main_color_name):
    main_h, _, _ = COLOR_HSV_MAP[main_color_name]
    comp_h = (main_h + 180) % 360
    h1 = (comp_h - 30) % 360
    h2 = (comp_h + 30) % 360
    group = [("分散互补1", h1, 0.42, 0.80), ("分散互补2", h2, 0.46, 0.74)]
    desc = "分散弱化互补色｜含蓄高级，轻熟通勤风"
    explain = "弱化撞色冲突，低调有质感，职场、轻熟气质穿搭适配"
    return group, desc, explain

def get_base_color_match():
    group = BASE_GRAY_COLORS + EARTH_COLORS
    desc = "基础万能基底搭配｜黑白灰+大地色系，所有彩色都能适配"
    explain = "基础色作为上衣/下装打底，彩色仅小面积点缀，零搭配翻车"
    return group, desc, explain

# 页面初始化
st.set_page_config(page_title="专业色卡衣物自动取色搭配工具（DeepSeek版）", page_icon="👗", layout="centered")
st.markdown("""
<style>
.stApp {background:#ffffff;}
.stButton>button {background:#222222;color:white;border-radius:6px;}
.area-box {border:1px solid #d8d8d8;padding:18px;border-radius:10px;margin-bottom:16px;}
.color-tag {padding:4px 10px;border-radius:4px;background:#eee;margin:4px 0;display:inline-block;}
</style>
""", unsafe_allow_html=True)
st.title("👗 专业标准色卡 上衣下装自动搭配工具（DeepSeek智能识别版）")
st.subheader("上传衣物使用DeepSeek AI智能识别主色调，匹配专业色卡，支持黑白灰/大地/全冷暖色系")
st.divider()

# 显示API配置状态
if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your_deepseek_api_key_here":
    st.success("✅ DeepSeek API已配置，将使用AI智能识别颜色")
else:
    st.warning("⚠️ DeepSeek API未配置，将使用备用机械式识别算法。请在.env文件中配置DEEPSEEK_API_KEY")

st.divider()

# 第一步 批量上传衣物
st.markdown("## 📸 第一步：批量上传上衣、下装（AI智能识别主色调）")
# 上衣区
st.markdown('<div class="area-box">', unsafe_allow_html=True)
st.subheader("🔴 上衣专区（支持多选批量上传）")
top_files = st.file_uploader("上传全部上衣图片", type=["jpg","png"], key="top_upload", accept_multiple_files=True)
top_imgs = []
top_name_list = []
top_color_cache = dict()
if top_files:
    for idx, f in enumerate(top_files):
        img = Image.open(f).convert("RGB")
        top_imgs.append(img)
        name = f"上衣{idx+1}"
        top_name_list.append(name)
        # 使用智能颜色识别
        main_color = smart_color_recognition(img)
        top_color_cache[name] = main_color
        st.image(img, width=160, caption=f"{name} | AI识别标准色：{main_color}")
st.markdown('</div>', unsafe_allow_html=True)

# 下装区
st.markdown('<div class="area-box">', unsafe_allow_html=True)
st.subheader("🟡 下装专区（支持多选批量上传）")
bottom_files = st.file_uploader("上传全部下装图片", type=["jpg","png"], key="bottom_upload", accept_multiple_files=True)
bottom_imgs = []
bottom_name_list = []
bottom_color_cache = dict()
if bottom_files:
    for idx, f in enumerate(bottom_files):
        img = Image.open(f).convert("RGB")
        bottom_imgs.append(img)
        name = f"下装{idx+1}"
        bottom_name_list.append(name)
        # 使用智能颜色识别
        main_color = smart_color_recognition(img)
        bottom_color_cache[name] = main_color
        st.image(img, width=160, caption=f"{name} | AI识别标准色：{main_color}")
st.markdown('</div>', unsafe_allow_html=True)

has_top_batch = len(top_imgs) > 0
has_bottom_batch = len(bottom_imgs) > 0
if has_top_batch and has_bottom_batch:
    st.success(f"✅ 已上传 {len(top_imgs)} 件上衣、{len(bottom_imgs)} 件下装，每件已通过AI智能识别标准主色，请下方各选一件搭配")
elif not has_top_batch:
    st.warning("请先上传上衣图片")
elif not has_bottom_batch:
    st.warning("请上传下装图片")
st.divider()

# 第二步 选择搭配单品
selected_top_img = None
selected_bottom_img = None
auto_main_color = ""
if has_top_batch and has_bottom_batch:
    st.markdown("## ✂ 第二步：各挑选一件，自动读取AI识别出的标准色彩")
    col1, col2 = st.columns(2)
    with col1:
        top_select_idx = st.selectbox("选择搭配上衣", range(len(top_name_list)), format_func=lambda x: top_name_list[x])
        sel_top_name = top_name_list[top_select_idx]
        selected_top_img = top_imgs[top_select_idx]
        top_recog_color = top_color_cache[sel_top_name]
        st.image(selected_top_img, width=220, caption=f"本次搭配上衣：{sel_top_name}")
        st.markdown(f'<span class="color-tag">上衣AI识别标准色：{top_recog_color}</span>', unsafe_allow_html=True)
    with col2:
        bottom_select_idx = st.selectbox("选择搭配下装", range(len(bottom_name_list)), format_func=lambda x: bottom_name_list[x])
        sel_bottom_name = bottom_name_list[bottom_select_idx]
        selected_bottom_img = bottom_imgs[bottom_select_idx]
        bottom_recog_color = bottom_color_cache[sel_bottom_name]
        st.image(selected_bottom_img, width=220, caption=f"本次搭配下装：{sel_bottom_name}")
        st.markdown(f'<span class="color-tag">下装AI识别标准色：{bottom_recog_color}</span>', unsafe_allow_html=True)
    auto_main_color = top_recog_color
    st.info(f"系统自动采用上衣AI识别标准色【{auto_main_color}】作为整套穿搭主色调，可下拉手动切换全部专业色卡色系")
    st.divider()
    
    # 第三步 色调选择（下拉框同步全部专业色卡颜色）
    st.markdown("## 🎨 第三步：确认穿搭主色调（完整覆盖专业色卡全部色系）")
    main_color_name = st.selectbox("整套穿搭标准主色调（自动填充AI识别色）",
                                    ALL_STD_COLOR_NAMES,
                                    index=ALL_STD_COLOR_NAMES.index(auto_main_color))
    
    # 色系冷暖提示（匹配你色卡文字释义）
    if main_color_name in BASE_GRAY_COLORS:
        st.info(f"【{main_color_name}】无彩色基底｜百搭万能，极简通勤、冷淡风穿搭，适配所有彩色单品")
        color_list, tip_desc, scene_explain = get_base_color_match()
    elif main_color_name in EARTH_COLORS:
        st.success(f"【{main_color_name}】大地暖色系｜温暖沉稳，日常通勤、复古温柔穿搭")
        mode = st.radio("选择配色方案", [
            "① 同色系深浅搭配｜温柔统一",
            "② 邻近柔和搭配｜休闲日常",
            "③ 基础色万能搭配｜不出错通勤",
            "④ 互补撞色搭配｜度假吸睛",
            "⑤ 三角复古搭配｜潮流穿搭",
            "⑥ 分散轻熟搭配｜低调高级"
        ])
        st.divider()
        if mode.startswith("①"):
            color_list, tip_desc, scene_explain = get_same_tone_match(main_color_name)
        elif mode.startswith("②"):
            color_list, tip_desc, scene_explain = get_adjacent_match(main_color_name)
        elif mode.startswith("③"):
            color_list, tip_desc, scene_explain = get_base_color_match()
        elif mode.startswith("④"):
            color_list, tip_desc, scene_explain = get_complement_match(main_color_name)
        elif mode.startswith("⑤"):
            color_list, tip_desc, scene_explain = get_triangle_match(main_color_name)
        else:
            color_list, tip_desc, scene_explain = get_split_complement(main_color_name)
    elif main_color_name in WARM_COLORS:
        st.success(f"【{main_color_name}】暖色｜热情温柔，约会、韩系少女穿搭")
        mode = st.radio("选择配色方案", [
            "① 同色系深浅搭配｜温柔统一",
            "② 邻近柔和搭配｜休闲日常",
            "③ 基础色万能搭配｜不出错通勤",
            "④ 互补撞色搭配｜度假吸睛",
            "⑤ 三角复古搭配｜潮流穿搭",
            "⑥ 分散轻熟搭配｜低调高级"
        ])
        st.divider()
        if mode.startswith("①"):
            color_list, tip_desc, scene_explain = get_same_tone_match(main_color_name)
        elif mode.startswith("②"):
            color_list, tip_desc, scene_explain = get_adjacent_match(main_color_name)
        elif mode.startswith("③"):
            color_list, tip_desc, scene_explain = get_base_color_match()
        elif mode.startswith("④"):
            color_list, tip_desc, scene_explain = get_complement_match(main_color_name)
        elif mode.startswith("⑤"):
            color_list, tip_desc, scene_explain = get_triangle_match(main_color_name)
        else:
            color_list, tip_desc, scene_explain = get_split_complement(main_color_name)
    else:
        st.info(f"【{main_color_name}】冷色系｜沉静清冷，春夏通勤、极简高级穿搭")
        mode = st.radio("选择配色方案", [
            "① 同色系深浅搭配｜温柔统一",
            "② 邻近柔和搭配｜休闲日常",
            "③ 基础色万能搭配｜不出错通勤",
            "④ 互补撞色搭配｜度假吸睛",
            "⑤ 三角复古搭配｜潮流穿搭",
            "⑥ 分散轻熟搭配｜低调高级"
        ])
        st.divider()
        if mode.startswith("①"):
            color_list, tip_desc, scene_explain = get_same_tone_match(main_color_name)
        elif mode.startswith("②"):
            color_list, tip_desc, scene_explain = get_adjacent_match(main_color_name)
        elif mode.startswith("③"):
            color_list, tip_desc, scene_explain = get_base_color_match()
        elif mode.startswith("④"):
            color_list, tip_desc, scene_explain = get_complement_match(main_color_name)
        elif mode.startswith("⑤"):
            color_list, tip_desc, scene_explain = get_triangle_match(main_color_name)
        else:
            color_list, tip_desc, scene_explain = get_split_complement(main_color_name)
    
    st.info(f"【当前上下装配色原理】{tip_desc}")
    st.markdown(f"【适配穿搭场景】{scene_explain}")
    st.divider()
    
    # 成套搭配文字讲解
    st.markdown("# 🤖 选中单件上衣+单件下装 专属成套搭配方案")
    st.markdown("### 60/30/10专业色彩分配规则（对标标准服装配色）")
    st.markdown("""
    - 60%主色：宽松长款上衣/阔腿下装，整套视觉核心（AI识别出的衣物主色调）
    - 30%辅助色：另一件单品使用配套调和色，上下形成层次过渡
    - 10%点缀色：包包、鞋子、项链、丝巾等小配饰，仅小面积提亮
    """)
    st.write(f"图片AI智能识别标准色：上衣{top_recog_color}、下装{bottom_recog_color}，全部匹配专业服装色卡生成协调搭配，无需自行配色")
    st.divider()
    
    # 专业色卡可视化展示+详细说明
    st.markdown("# 🎨 本次搭配专用标准色卡（对标你提供的全套服装色卡）")
    st.markdown("""
    ### 色卡分三大区域，对照专业服装色彩逻辑使用
    1. 左侧【无黑白灰基底区】：纯白~黑色、米白，万能打底，任何彩色上衣/下装都能搭配
    2. 中间【大地百搭色系区】：燕麦、驼色、卡其、棕系，温柔复古通勤万能色
    3. 右侧【成套搭配三色区】：严格60%主色 / 30%辅助色 / 10%点缀色，直接照搬穿搭比例
    """)
    st.info("穿搭示范：驼色宽松上衣(60%主色) + 80%灰阔腿裤(30%基底辅助) + 藏蓝小包(10%点缀)")
    st.divider()
    
    # 绘制专业分区色卡
    fig, ax = plt.subplots(figsize=(12, 2.2))
    ax.axis("off")
    
    # 1. 左侧无黑白灰基底色块
    base_gray_info = [
        ("纯白", "#fcfcfc"), ("20%灰", "#d0d0d0"), ("40%灰", "#a0a0a0"),
        ("60%灰", "#686868"), ("80%灰", "#383838"), ("黑色", "#121212"), ("米白", "#f8f3e6")
    ]
    base_w = 0.095
    for idx, (name, hex_c) in enumerate(base_gray_info):
        x = 0.02 + idx * base_w
        rect = plt.Rectangle((x, 0.12), base_w*0.92, 0.76, color=hex_c)
        ax.add_patch(rect)
        text_color = "black" if hex_c in ["#fcfcfc","#f8f3e6","#d0d0d0"] else "white"
        ax.text(x + base_w/2, 0.5, name, ha="center", fontsize=9, color=text_color)
    ax.text(0.02+base_w*3, 0.92, "【无彩色基底】万能打底", fontsize=11, ha="center")
    
    # 2. 中间大地色系色块
    earth_info = [("燕麦", "#f2e8d5"), ("驼色", "#d4b88c"), ("卡其", "#c9b07a"), ("深棕", "#6b4c32")]
    earth_start_x = 0.72
    earth_w = 0.062
    for idx, (name, hex_c) in enumerate(earth_info):
        x = earth_start_x + idx * earth_w
        rect = plt.Rectangle((x, 0.12), earth_w*0.9, 0.76, color=hex_c)
        ax.add_patch(rect)
        ax.text(x + earth_w/2, 0.5, name, ha="center", fontsize=9, color="black")
    ax.text(earth_start_x+earth_w*2, 0.92, "【大地百搭色系】", fontsize=11, ha="center")
    
    # 3. 右侧60/30/10成套三色区
    seg_start_x = 0.42
    seg_w = 0.13
    seg_labels = ["60%主色", "30%辅助色", "10%点缀色"]
    if main_color_name in BASE_GRAY_COLORS:
        seg_color_hex = ["#909090", "#b8a999", "#304060"]
        seg_names = ["基底主色", "大地辅助", "冷调点缀"]
    else:
        h_main, s_main, v_main = COLOR_HSV_MAP[main_color_name]
        seg_color_hex = [
            hsv_to_hex(h_main, s_main, v_main),
            hsv_to_hex((h_main+40)%360, 0.40, 0.85),
            hsv_to_hex((h_main+180)%360, 0.55, 0.72)
        ]
        seg_names = [main_color_name, "辅助色", "点缀色"]
    
    for idx, (label, hex_c, name) in enumerate(zip(seg_labels, seg_color_hex, seg_names)):
        x = seg_start_x + idx * seg_w
        rect = plt.Rectangle((x, 0.12), seg_w*0.9, 0.76, color=hex_c)
        ax.add_patch(rect)
        text_color = "black" if hex_c in ["#fcfcfc","#f8f3e6","#d0d0d0","#f2e8d5","#d4b88c","#c9b07a"] else "white"
        ax.text(x + seg_w/2, 0.5, name, ha="center", fontsize=10, color=text_color)
        ax.text(x + seg_w/2, 0.02, label, ha="center", fontsize=9)
    ax.text(seg_start_x+seg_w*1.5, 0.92, "【成套搭配三色区】", fontsize=11, ha="center")
    
    st.pyplot(fig)
    st.divider()
    
    # 导出搭配方案图（仅展示选中一件上衣一件下装）
    if selected_top_img and selected_bottom_img:
        st.markdown("## 导出本次搭配方案图")
        export_btn = st.button("生成搭配方案图（包含上衣+下装+色卡）")
        if export_btn:
            # 创建导出图，使用子图布局
            export_fig = plt.figure(figsize=(16, 6))
            
            # 上衣子图（左侧）
            ax_top = export_fig.add_subplot(1, 3, 1)
            ax_top.axis("off")
            ax_top.set_title(f"上衣：{sel_top_name}\nAI识别色：{top_recog_color}", fontsize=12)
            ax_top.imshow(selected_top_img)
            
            # 下装子图（中间）
            ax_bottom = export_fig.add_subplot(1, 3, 2)
            ax_bottom.axis("off")
            ax_bottom.set_title(f"下装：{sel_bottom_name}\nAI识别色：{bottom_recog_color}", fontsize=12)
            ax_bottom.imshow(selected_bottom_img)
            
            # 色卡子图（右侧）
            ax_color = export_fig.add_subplot(1, 3, 3)
            ax_color.axis("off")
            ax_color.set_title("专业色卡", fontsize=14)
            
            base_gray_info = [
                ("纯白", "#fcfcfc"), ("20%灰", "#d0d0d0"), ("40%灰", "#a0a0a0"),
                ("60%灰", "#686868"), ("80%灰", "#383838"), ("黑色", "#121212"), ("米白", "#f8f3e6")
            ]
            earth_info = [("燕麦", "#f2e8d5"), ("驼色", "#d4b88c"), ("卡其", "#c9b07a"), ("深棕", "#6b4c32")]
            
            # 绘制黑白灰基底色系
            ax_color.text(0.5, 0.92, "【无彩色基底】", fontsize=10, ha="center")
            base_w = 0.12
            for idx, (name, hex_c) in enumerate(base_gray_info):
                x = 0.05 + idx * base_w
                rect = plt.Rectangle((x, 0.7), base_w*0.9, 0.18, color=hex_c)
                ax_color.add_patch(rect)
                text_color = "black" if hex_c in ["#fcfcfc","#f8f3e6","#d0d0d0"] else "white"
                ax_color.text(x + base_w/2, 0.79, name, ha="center", fontsize=8, color=text_color)
            
            # 绘制大地色系
            ax_color.text(0.5, 0.58, "【大地百搭色系】", fontsize=10, ha="center")
            earth_w = 0.18
            for idx, (name, hex_c) in enumerate(earth_info):
                x = 0.1 + idx * earth_w
                rect = plt.Rectangle((x, 0.38), earth_w*0.9, 0.18, color=hex_c)
                ax_color.add_patch(rect)
                ax_color.text(x + earth_w/2, 0.47, name, ha="center", fontsize=8, color="black")
            
            # 绘制60/30/10配色区
            ax_color.text(0.5, 0.2, "【60/30/10配色】", fontsize=10, ha="center")
            if main_color_name in BASE_GRAY_COLORS:
                seg_color_hex = ["#909090", "#b8a999", "#304060"]
                seg_names = ["基底主色", "大地辅助", "冷调点缀"]
            else:
                h_main, s_main, v_main = COLOR_HSV_MAP[main_color_name]
                seg_color_hex = [
                    hsv_to_hex(h_main, s_main, v_main),
                    hsv_to_hex((h_main+40)%360, 0.40, 0.85),
                    hsv_to_hex((h_main+180)%360, 0.55, 0.72)
                ]
                seg_names = [main_color_name, "辅助色", "点缀色"]
            
            seg_w = 0.28
            for idx, (name, hex_c) in enumerate(zip(seg_names, seg_color_hex)):
                x = 0.07 + idx * seg_w
                rect = plt.Rectangle((x, 0.02), seg_w*0.9, 0.15, color=hex_c)
                ax_color.add_patch(rect)
                text_color = "black" if hex_c in ["#fcfcfc","#f8f3e6","#d0d0d0","#f2e8d5","#d4b88c","#c9b07a"] else "white"
                ax_color.text(x + seg_w/2, 0.095, name, ha="center", fontsize=9, color=text_color)
            
            plt.tight_layout()
            st.pyplot(export_fig)
            
            # 导出按钮
            buf = io.BytesIO()
            export_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                label="下载搭配方案图",
                data=buf,
                file_name=f"搭配方案_{sel_top_name}_{sel_bottom_name}.png",
                mime="image/png"
            )

st.caption("升级优化：1.接入DeepSeek AI智能识别颜色，更准确更智能 2.内置全套专业服装色卡数据库 3.支持黑白灰/大地/冷暖全色系 4.保留备用机械式识别算法 5.色卡分三大专业分区易懂 6.导出图仅展示选中一件上衣一件下装")