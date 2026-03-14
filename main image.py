import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import cv2
import matplotlib
matplotlib.use("Agg")  # Safe plotting in VS Code / headless
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from skimage.metrics import structural_similarity as ssim_metric
from fpdf import FPDF  # pip install fpdf

# -------------------------
# Image Utilities
# -------------------------
def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

def save_image(arr, path):
    Image.fromarray(arr).save(path)

def rgb_to_lab(arr):
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)

def lab_to_rgb(arr):
    return cv2.cvtColor(np.clip(arr,0,255).astype(np.uint8), cv2.COLOR_LAB2RGB)

# -------------------------
# Pixel Sampling
# -------------------------
def sample_pixels(arr, max_samples=50000, seed=42):
    flat = arr.reshape(-1,3)
    if flat.shape[0] <= max_samples:
        return flat.astype(np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat.shape[0], max_samples, replace=False)
    return flat[idx].astype(np.float32)

# -------------------------
# Metrics
# -------------------------
def evaluate_metrics(original, compressed):
    original = original.astype(np.uint8)
    compressed = compressed.astype(np.uint8)

    mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32))**2)
    psnr = float("inf") if mse==0 else 10*np.log10((255.0**2)/mse)

    h,w = original.shape[:2]
    win = min(7,h,w)
    if win % 2 == 0: win -= 1
    if win < 3: win = 3
    s = ssim_metric(original, compressed, data_range=255, win_size=win, channel_axis=-1)
    return psnr, s

# -------------------------
# Compression
# -------------------------
def compress_image(img, k, colorspace="lab", seed=42):
    if colorspace=="lab":
        img_cs = rgb_to_lab(img)
    else:
        img_cs = img.astype(np.float32)

    samples = sample_pixels(img_cs)
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=seed)
    start_time = time.time()
    kmeans.fit(samples)
    labels = kmeans.predict(img_cs.reshape(-1,3))
    compressed = kmeans.cluster_centers_[labels].reshape(img_cs.shape)
    elapsed = time.time() - start_time

    if colorspace=="lab":
        compressed = lab_to_rgb(compressed)

    return compressed.astype(np.uint8), kmeans.cluster_centers_, elapsed

# -------------------------
# Plotting (Only Average Plots)
# -------------------------
def save_metric_plots(all_results, out_dir, per_image=False):
    os.makedirs(out_dir, exist_ok=True)

    def safe_save_plot(x, y, xlabel, ylabel, title, filename):
        try:
            plt.figure(figsize=(6,4), dpi=80)
            plt.plot(x, y, marker="o")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, filename))
            plt.close()
        except Exception as e:
            print(f" Could not save plot {filename}: {e}")
            plt.close()

    # Average plots only
    avg_results = {}
    for k in set(r["k"] for res in all_results.values() for r in res):
        psnrs = [r["psnr"] for res in all_results.values() for r in res if r["k"]==k]
        ssims = [r["ssim"] for res in all_results.values() for r in res if r["k"]==k]
        times = [r["time"] for res in all_results.values() for r in res if r["k"]==k]
        avg_results[k] = {"psnr": np.mean(psnrs), "ssim": np.mean(ssims), "time": np.mean(times)}

    ks = sorted(avg_results.keys())
    safe_save_plot(ks, [avg_results[k]["psnr"] for k in ks], "K", "PSNR (dB)", "Average PSNR vs K", "avg_psnr.png")
    safe_save_plot(ks, [avg_results[k]["ssim"] for k in ks], "K", "SSIM", "Average SSIM vs K", "avg_ssim.png")
    safe_save_plot(ks, [avg_results[k]["time"] for k in ks], "K", "Time (s)", "Average Time vs K", "avg_time.png")

# -------------------------
# PDF Report with Side-by-Side
# -------------------------
def generate_pdf_report(all_results, input_folder, output_folder, max_width=800):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0,10,"K-Means Image Compression Report", ln=True, align="C")
    pdf.set_font("Arial", '', 12)
    pdf.ln(5)
    pdf.cell(0,8,f"Total images: {len(all_results)}", ln=True)
    pdf.cell(0,8,"Generated with MiniBatchKMeans and pixel sampling", ln=True)
    pdf.ln(5)

    # Add overall plots
    for fname in ["avg_psnr.png","avg_ssim.png","avg_time.png"]:
        path = os.path.join(output_folder,fname)
        if os.path.exists(path):
            try:
                pdf.image(os.path.normpath(path), w=180)
                pdf.ln(5)
            except Exception as e:
                print(f"Could not add plot {path} to PDF: {e}")

    # Add each image results
    for img_name, results in all_results.items():
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0,10,f"Image: {img_name}", ln=True)
        pdf.set_font("Arial", '', 12)

        for r in results:
            k = r["k"]
            psnr_val, ssim_val, elapsed = r["psnr"], r["ssim"], r["time"]
            comp_img_path = os.path.join(output_folder,f"{img_name}_k{k}.png")
            
            # Find original image path
            orig_img_path = None
            for ext in ["jpg","png","jpeg","bmp"]:
                tmp = os.path.join(input_folder,f"{img_name}.{ext}")
                if os.path.exists(tmp):
                    orig_img_path = tmp
                    break
            if orig_img_path is None or not os.path.exists(comp_img_path):
                continue

            # Side-by-side
            try:
                orig = Image.open(orig_img_path).convert("RGB")
                comp = Image.open(comp_img_path).convert("RGB")
                max_h = max(orig.height, comp.height)
                scale_orig = max_h/orig.height
                scale_comp = max_h/comp.height
                orig = orig.resize((int(orig.width*scale_orig), max_h))
                comp = comp.resize((int(comp.width*scale_comp), max_h))
                side_img = Image.new("RGB",(orig.width+comp.width, max_h),(255,255,255))
                side_img.paste(orig,(0,0))
                side_img.paste(comp,(orig.width,0))
                # Resize for PDF
                if side_img.width > max_width:
                    scale = max_width/side_img.width
                    side_img = side_img.resize((max_width,int(side_img.height*scale)))
                tmp_path = os.path.join(output_folder,f"{img_name}_side_k{k}.png")
                side_img.save(tmp_path)
                pdf.cell(0,8,f"K={k} | PSNR={psnr_val:.2f} | SSIM={ssim_val:.4f} | Time={elapsed:.2f}s", ln=True)
                pdf.image(os.path.normpath(tmp_path), w=180)
                pdf.ln(5)
            except Exception as e:
                print(f"Could not add side image for {img_name} K={k}: {e}")

    pdf_file = os.path.join(output_folder,"KMeans_Compression_Report.pdf")
    pdf.output(pdf_file)
    print(f"\n✔ PDF report generated: {pdf_file}")

# -------------------------
# Folder Processing
# -------------------------
def process_folder_multiK(input_folder, output_folder, k_values, colorspace="lab", seed=42):
    os.makedirs(output_folder, exist_ok=True)
    images = [os.path.splitext(f)[0] for f in os.listdir(input_folder) if f.lower().endswith((".jpg",".png",".jpeg",".bmp"))]
    all_results = {}

    for img_name in tqdm(images, desc="Processing images"):
        path = None
        for ext in ["jpg","png","jpeg","bmp"]:
            tmp = os.path.join(input_folder,f"{img_name}.{ext}")
            if os.path.exists(tmp):
                path = tmp
                break
        if path is None:
            continue
        img = load_image(path)
        results = []

        for k in k_values:
            compressed, palette, elapsed = compress_image(img,k,colorspace,seed)
            psnr_val, ssim_val = evaluate_metrics(img, compressed)
            save_image(compressed, os.path.join(output_folder,f"{img_name}_k{k}.png"))
            np.save(os.path.join(output_folder,f"{img_name}_palette_k{k}.npy"), palette)
            results.append({"k":k,"psnr":psnr_val,"ssim":ssim_val,"time":elapsed})

        all_results[img_name] = results

    save_metric_plots(all_results, output_folder)
    generate_pdf_report(all_results, input_folder, output_folder)
    print(f"\n Done. All metrics, side-by-side images, plots, and PDF saved in {output_folder}")

# -------------------------
# Main
# -------------------------
def main():
    import sys
    if len(sys.argv)==1:
        input_folder = r"C:\Users\DELL\Desktop\ML prOJECT\daisy"
        output_folder = r"C:\Users\DELL\Desktop\ML prOJECT\output"
        k_values = [2,4,8,16,32]
        colorspace = "lab"
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_folder", required=True)
        parser.add_argument("--out_folder", default="output")
        parser.add_argument("--k_values", nargs="+", type=int, required=True)
        parser.add_argument("--colorspace", choices=["rgb","lab"], default="lab")
        args = parser.parse_args()
        input_folder = args.input_folder
        output_folder = args.out_folder
        k_values = args.k_values
        colorspace = args.colorspace

    process_folder_multiK(input_folder, output_folder, k_values, colorspace)

if __name__=="__main__":
    main()
