import streamlit as st
import json
import os
from PIL import Image, ImageDraw
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_json(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"读取JSON文件失败: {e}")
        return None

def draw_bbox_on_image(image_path, bbox):
    """在图片上绘制BBox"""
    if not image_path:
        return None, "图片路径为空"
    
    if not os.path.exists(image_path):
        return None, f"图片文件未找到: {image_path}"
    
    try:
        # 使用 with 语句确保文件句柄被正确关闭
        with Image.open(image_path) as f:
            image = f.convert("RGB")
            
        width, height = image.size
        draw = ImageDraw.Draw(image)
        
        if bbox and len(bbox) == 4:
            abs_xmin = (bbox[0] / 1000.0) * width
            abs_ymin = (bbox[1] / 1000.0) * height
            abs_xmax = (bbox[2] / 1000.0) * width
            abs_ymax = (bbox[3] / 1000.0) * height
            
            draw.rectangle(
                [abs_xmin, abs_ymin, abs_xmax, abs_ymax], 
                outline="red", 
                width=max(3, int(min(width, height) * 0.005))
            )
        return image, None
    except Exception as e:
        return None, f"处理图片时出错: {e}"

def main():
    st.set_page_config(layout="wide", page_title="Bad Case Analysis")
    st.title("🐞 Bad Case 分析工具")
    
    # --- 侧边栏：配置 ---
    st.sidebar.header("📂 数据加载")
    
    # 默认路径
    default_path = os.path.join(os.getcwd(), "output", "bad_cases", "retrieval_bad_cases.json")
    if not os.path.exists(default_path):
        default_path = os.getcwd()

    input_path = st.sidebar.text_input("文件路径或目录 (JSON):", value=default_path)
    
    json_files = []
    
    # 逻辑判断：是文件还是目录
    if os.path.isfile(input_path):
        if input_path.endswith(".json"):
            json_files = [input_path]
        else:
            st.sidebar.error("请选择一个 .json 文件")
            st.stop()
    elif os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))
        json_files.sort()
        if not json_files:
            st.sidebar.warning("该目录下没有找到 JSON 文件")
            st.stop()
    else:
        st.sidebar.error(f"路径不存在: {input_path}")
        st.stop()

    # --- 状态管理 ---
    if 'file_index' not in st.session_state:
        st.session_state.file_index = 0

    # 1. 文件选择器（如果是单文件，只有一个选项）
    file_selector = st.sidebar.selectbox(
        "选择文件:", 
        json_files,
        format_func=lambda x: os.path.basename(x)
    )
    
    # 每次切换文件时，重置索引（可选，取决于用户习惯，这里保持状态可能更好，或者重置为0）
    # 为了简单起见，如果文件名变了，可以考虑重置，但 Streamlit 的 selectbox 改变会自动重刷页面
    
    data_list = load_json(file_selector)
    if not isinstance(data_list, list):
        # 兼容旧格式或单样本格式
        data_list = [data_list] if data_list else []
    
    if not data_list:
        st.warning(f"文件 {os.path.basename(file_selector)} 为空或格式错误")
        st.stop()

    # --- 样本导航 ---
    total_samples = len(data_list)
    st.sidebar.subheader(f"样本列表 ({total_samples})")
    
    col_prev, col_info, col_next = st.sidebar.columns([1, 2, 1])
    
    # 翻页逻辑
    with col_prev:
        if st.button("⬅️") and st.session_state.file_index > 0:
            st.session_state.file_index -= 1
    with col_next:
        if st.button("➡️") and st.session_state.file_index < total_samples - 1:
            st.session_state.file_index += 1
            
    with col_info:
        st.markdown(f"<div style='text-align: center; line-height: 2.2;'>{st.session_state.file_index + 1} / {total_samples}</div>", unsafe_allow_html=True)
        
    # 滑块快速跳转
    if total_samples > 1:
        new_index = st.sidebar.slider("跳转索引:", 1, total_samples, st.session_state.file_index + 1) - 1
        st.session_state.file_index = new_index

    # 确保索引不越界（切换文件后可能发生）
    if st.session_state.file_index >= total_samples:
        st.session_state.file_index = 0
        
    current_data = data_list[st.session_state.file_index]
    
    st.sidebar.divider()
    
    # --- 内容展示 ---
    if current_data:
        metrics = current_data.get("metrics", {})
        
        # 1. 顶部状态栏：Bad Case 类型提示
        # 兼容扁平化 key 和嵌套 key
        recall = metrics.get('page_recall', metrics.get('page', {}).get('recall', 0.0))
        model_eval = metrics.get('model_eval', 0.0)
        
        is_retrieval_fail = recall < 1.0
        is_gen_fail = model_eval < 0.5
        
        status_cols = st.columns([1, 3])
        with status_cols[0]:
            if is_retrieval_fail:
                st.error("❌ Retrieval Failure")
            elif is_gen_fail:
                st.error("❌ Generation Failure")
            else:
                st.success("✅ Passed")
                
        # 2. 指标展示 (Metrics)
        with st.expander("📊 评估指标详情 (Metrics)", expanded=True):
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Model Score", f"{model_eval:.2f}")
            with m2:
                st.metric("Page Recall", f"{recall:.2%}")
            with m3:
                prec = metrics.get('page_precision', metrics.get('page', {}).get('precision', 0.0))
                st.metric("Page Precision", f"{prec:.2%}")
            with m4:
                gold_pages_count = len(current_data.get('gold_pages', []))
                st.metric("Gold Pages Count", gold_pages_count)

        # 3. 问答对比
        st.subheader("📝 Q&A Analysis")
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**Question (QID: {current_data.get('qid')})**\n\n{current_data.get('query', 'N/A')}")
            st.markdown(f"**Doc Source:** `{current_data.get('doc_source', 'N/A')}`")
        with c2:
            st.success(f"**Gold Answer:**\n\n{current_data.get('gold_answer', 'N/A')}")
            st.warning(f"**Model Answer:**\n\n{current_data.get('final_answer', 'N/A')}")
            
        st.divider()

        # 4. 检索证据展示
        st.subheader("🔍 Retrieved Evidence Analysis")
        
        if current_data.get('gold_pages'):
            st.markdown(f"**Correct Gold Pages:** `{current_data.get('gold_pages')}`")

        retrieved = current_data.get('retrieved_elements', [])
        if not retrieved:
            st.write("No elements retrieved.")
        
        for i, elem in enumerate(retrieved):
            # 简单的命中判断逻辑
            is_hit = False
            gold_pages = current_data.get('gold_pages', [])
            page_path = elem.get('corpus_path', '')
            if page_path and gold_pages:
                page_name = os.path.basename(page_path)
                # 模糊匹配：只要 gold_page 字符串出现在路径中就算命中
                if any(str(g) in page_name for g in gold_pages):
                    is_hit = True
            
            title_emoji = "✅ Hit" if is_hit else "📄"
            with st.container():
                st.markdown(f"#### {title_emoji} Evidence #{i+1}")
                col_text, col_img = st.columns([1, 1])
                
                with col_text:
                    content_preview = elem.get('content', '')
                    st.markdown("**Content**\n\n" +  content_preview)
                    
                    # Metadata 展示
                    meta_show = {k:v for k,v in elem.items() if k != 'content'}
                    with st.expander("Metadata"):
                        st.json(meta_show)
                
                with col_img:
                    path = elem.get('corpus_path') or elem.get('crop_path')
                    if path:
                        # 尝试加载图片
                        img, err = draw_bbox_on_image(path, elem.get('bbox'))
                        if img: 
                            # 核心修复：转为 numpy
                            st.image(
                                np.array(img), 
                                caption=f"File: {os.path.basename(path)}",
                            )
                        else: 
                            st.error(f"Image Load Error: {err}")
                st.divider()

if __name__ == "__main__":
    main()