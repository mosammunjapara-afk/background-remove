import os, io, uuid, json, base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance, ImageChops
import numpy as np

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///autolens.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

db = SQLAlchemy(app)

# ─── Models ───
class CarImage(db.Model):
    id = db.Column(db.String(50), primary_key=True)
    filename = db.Column(db.String(255))
    car_name = db.Column(db.String(255))
    original_path = db.Column(db.String(500))
    nobg_path = db.Column(db.String(500))
    processed_path = db.Column(db.String(500))
    status = db.Column(db.String(20), default='uploaded')
    in_gallery = db.Column(db.Boolean, default=False)
    bg_removal_method = db.Column(db.String(50), default='none')  # NEW: track method used
    bg_removal_quality = db.Column(db.String(20), default='standard')  # NEW: track quality
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class CreditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    action = db.Column(db.String(100))
    cost = db.Column(db.Integer, default=0)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()
    # Always create required folders at startup, not just in __main__
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

BACKGROUNDS = {
    'studio': [
        {'id': 'studio_white', 'name': 'Pure White', 'color': '#FFFFFF'},
        {'id': 'studio_grey', 'name': 'Studio Grey', 'color': '#E8E8E8'},
        {'id': 'studio_black', 'name': 'Midnight Black', 'color': '#1A1A1A'},
        {'id': 'studio_blue', 'name': 'Steel Blue', 'color': '#1E3A5F'},
        {'id': 'studio_gradient', 'name': 'Gradient Fog', 'color': '#C8D2E6'},
    ],
    'outdoor': [
        {'id': 'outdoor_road', 'name': 'Open Road', 'color': '#4A7C59'},
        {'id': 'outdoor_mountain', 'name': 'Mountain Pass', 'color': '#5B7FA6'},
        {'id': 'outdoor_sunset', 'name': 'Golden Sunset', 'color': '#FF6B35'},
        {'id': 'outdoor_city', 'name': 'City Night', 'color': '#2C1654'},
        {'id': 'outdoor_showroom', 'name': 'Showroom Floor', 'color': '#C0C0C0'},
    ]
}

SWATCHES = ['#FFFFFF','#E8E8E8','#1A1A1A','#1E3A5F','#C8D2E6','#4A7C59',
            '#5B7FA6','#FF6B35','#2C1654','#C0C0C0','#FF0000','#00FF00',
            '#0000FF','#FFD700','#FF1493','#00CED1','#8B4513','#FF4500']

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# ═══════════════════════════════════════════════════════════
#  CAR-SPECIFIC BACKGROUND REMOVAL ENGINE
# ═══════════════════════════════════════════════════════════

def remove_bg_rembg(filepath, quality='standard'):
    """
    Primary AI engine using rembg with car-optimised models.
    quality: 'draft' | 'standard' | 'high' | 'ultra'
    """
    try:
        from rembg import remove, new_session

        # Car-optimised model selection:
        # isnet-general-use & u2net are best for vehicles
        # NEVER use u2net_human_seg for cars
        model_map = {
            'draft':    'u2netp',             # Fast, decent for cars
            'standard': 'u2net',              # Best balance for cars
            'high':     'isnet-general-use',  # Excellent for vehicles
            'ultra':    'isnet-general-use',  # Best available
        }
        model = model_map.get(quality, 'u2net')

        try:
            session = new_session(model)
            with open(filepath, 'rb') as f:
                img_bytes = f.read()
            result = remove(img_bytes, session=session,
                          alpha_matting=True,
                          alpha_matting_foreground_threshold=240,
                          alpha_matting_background_threshold=10,
                          alpha_matting_erode_size=10)
        except Exception:
            with open(filepath, 'rb') as f:
                result = remove(f.read(), alpha_matting=True)

        img = Image.open(io.BytesIO(result)).convert('RGBA')
        # Always refine edges for cars
        img = refine_edges_car(img)
        return img, 'rembg_' + quality

    except ImportError:
        return None, None
    except Exception as e:
        print(f"rembg error: {e}")
        return None, None


def remove_bg_car_opencv(filepath, quality='standard'):
    """
    Car-specific OpenCV background removal.
    Uses multi-pass GrabCut + contour cleanup + alpha matting.
    Works well even without rembg.
    """
    try:
        import cv2

        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            raise ValueError("Cannot read image")

        h, w = img_bgr.shape[:2]

        # ── Step 1: Detect approximate car region using edges & saliency ──
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edges to find object boundary
        edges = cv2.Canny(blurred, 20, 80)
        kernel_d = np.ones((12, 12), np.uint8)
        edges_thick = cv2.dilate(edges, kernel_d, iterations=2)

        # ── Step 2: GrabCut pass 1 — wide rect ──
        margin_x = max(int(w * 0.02), 5)
        margin_y = max(int(h * 0.02), 5)
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

        mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(img_bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

        # ── Step 3: Lock corners as definite background ──
        cs = max(int(min(w, h) * 0.06), 10)
        mask[:cs, :cs]   = cv2.GC_BGD
        mask[:cs, -cs:]  = cv2.GC_BGD
        mask[-cs:, :cs]  = cv2.GC_BGD
        mask[-cs:, -cs:] = cv2.GC_BGD

        # Also mark thin strips along all 4 edges as background
        strip = max(int(min(w, h) * 0.01), 3)
        mask[:strip, :]  = cv2.GC_BGD
        mask[-strip:, :] = cv2.GC_BGD
        mask[:, :strip]  = cv2.GC_BGD
        mask[:, -strip:] = cv2.GC_BGD

        # ── Step 4: GrabCut pass 2 — refine ──
        cv2.grabCut(img_bgr, mask, rect, bgd, fgd, 8, cv2.GC_EVAL)

        fg_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        # ── Step 5: Morphological cleanup (car-specific sizes) ──
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        k_fill  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))

        # Close gaps (wheels, gaps under car)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k_close, iterations=4)
        # Remove small specks
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  k_open,  iterations=1)
        # Fill interior holes (windows, grille)
        fg_filled = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, k_fill, iterations=2)
        fg_filled = cv2.morphologyEx(fg_filled, cv2.MORPH_ERODE, k_fill, iterations=2)

        # ── Step 6: Keep only the largest contour (the car body) ──
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Sort by area, keep top 3 (car body + wheels might be separate)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            max_area = cv2.contourArea(contours[0])
            clean_mask = np.zeros_like(fg_mask)
            for c in contours:
                if cv2.contourArea(c) > max_area * 0.05:  # keep contours >5% of largest
                    cv2.drawContours(clean_mask, [c], -1, 255, -1)
            fg_mask = clean_mask

        # Fill holes inside the car mask (windows appear as holes)
        fg_mask = fill_holes(fg_mask)

        # ── Step 7: Color-aware alpha matting at edges ──
        # Create soft transition zone at edges using distance transform
        dist = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)
        dist_inv = cv2.distanceTransform(255 - fg_mask, cv2.DIST_L2, 5)

        feather_px = max(3, int(min(w, h) * 0.005))
        alpha = np.where(
            fg_mask == 255,
            np.clip(255 * dist / (feather_px + 0.001), 0, 255),
            0
        ).astype(np.uint8)

        # Final smooth
        alpha = cv2.GaussianBlur(alpha, (5, 5), 1.5)

        # ── Step 8: Build RGBA PIL image ──
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).convert('RGBA')
        alpha_pil = Image.fromarray(alpha).convert('L')
        pil_img.putalpha(alpha_pil)

        return pil_img, 'opencv_car'

    except ImportError:
        return None, None
    except Exception as e:
        print(f"OpenCV car engine error: {e}")
        import traceback; traceback.print_exc()
        return None, None


def fill_holes(mask):
    """Fill interior holes in a binary mask (e.g. car windows)."""
    try:
        import cv2
        # Flood fill from border
        h, w = mask.shape
        flood = mask.copy()
        flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, flood_fill_mask, (0, 0), 255)
        flood_inv = cv2.bitwise_not(flood)
        filled = mask | flood_inv
        return filled
    except Exception:
        return mask


def remove_bg_color_smart(filepath, threshold=40):
    """
    Smart color-keying for studio cars on solid backgrounds.
    Uses flood-fill from edges + statistical background detection.
    """
    try:
        import cv2

        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            raise ValueError()
        h, w = img_bgr.shape[:2]

        # Sample background from edge strips
        top    = img_bgr[:int(h*0.08), :]
        bottom = img_bgr[int(h*0.92):, :]
        left   = img_bgr[:, :int(w*0.08)]
        right  = img_bgr[:, int(w*0.92):]
        edge_pixels = np.vstack([
            top.reshape(-1, 3), bottom.reshape(-1, 3),
            left.reshape(-1, 3), right.reshape(-1, 3)
        ])

        # Mean background color
        bg_color = edge_pixels.mean(axis=0)  # BGR

        # Distance from background
        diff = img_bgr.astype(np.float32) - bg_color
        dist_map = np.sqrt((diff ** 2).sum(axis=2))

        # Threshold
        alpha = np.clip((dist_map - threshold) / threshold * 255, 0, 255).astype(np.uint8)

        # Morphological cleanup
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, k, iterations=3)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN,  k, iterations=1)
        alpha = fill_holes(alpha)

        # Smooth
        alpha = cv2.GaussianBlur(alpha, (7, 7), 2)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).convert('RGBA')
        pil_img.putalpha(Image.fromarray(alpha))
        return pil_img, 'color_smart'

    except Exception as e:
        print(f"color_smart error: {e}")
        return None, None


def refine_edges_car(img_rgba, radius=1):
    """Refine cutout edges — remove semi-transparent fringing."""
    r, g, b, a = img_rgba.split()
    a_arr = np.array(a, dtype=np.float32)
    # Threshold soft edges: push near-opaque pixels to fully opaque
    a_arr = np.where(a_arr > 200, 255, np.where(a_arr < 30, 0, a_arr))
    a_clean = Image.fromarray(a_arr.astype(np.uint8))
    if radius > 0:
        a_clean = a_clean.filter(ImageFilter.GaussianBlur(radius))
    return Image.merge('RGBA', (r, g, b, a_clean))


def refine_edges(img_rgba, radius=2):
    """Generic edge refinement (kept for API compatibility)."""
    return refine_edges_car(img_rgba, radius)


def despill(img_rgba, bg_color_hint=(255, 255, 255)):
    """Remove background color spill from car edges."""
    r, g, b, a = img_rgba.split()
    r_arr = np.array(r, dtype=np.float32)
    g_arr = np.array(g, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    a_arr = np.array(a, dtype=np.float32) / 255.0
    br, bg_c, bb = bg_color_hint
    dominant = max(br, bg_c, bb)
    if dominant == br:
        r_arr = np.clip(r_arr - (g_arr + b_arr) / 2 * (1 - a_arr) * 0.3, 0, 255)
    elif dominant == bg_c:
        g_arr = np.clip(g_arr - (r_arr + b_arr) / 2 * (1 - a_arr) * 0.3, 0, 255)
    else:
        b_arr = np.clip(b_arr - (r_arr + g_arr) / 2 * (1 - a_arr) * 0.3, 0, 255)
    return Image.merge('RGBA', (
        Image.fromarray(r_arr.astype(np.uint8)),
        Image.fromarray(g_arr.astype(np.uint8)),
        Image.fromarray(b_arr.astype(np.uint8)), a
    ))


def remove_bg_ai(filepath, quality='standard', engine='auto', despill_enable=False):
    """
    Master car background removal dispatcher.
    Priority: rembg AI → OpenCV GrabCut (car-tuned) → Smart Color Key

    engine: 'auto' | 'rembg' | 'grabcut' | 'color_key'
    quality: 'draft' | 'standard' | 'high' | 'ultra'
    """
    result_img = None
    method_used = 'fallback'

    # 1. Try rembg AI (best quality)
    if engine in ('auto', 'rembg'):
        result_img, method_used = remove_bg_rembg(filepath, quality)

    # 2. Try car-optimised OpenCV GrabCut
    if result_img is None and engine in ('auto', 'grabcut'):
        result_img, method_used = remove_bg_car_opencv(filepath, quality)

    # 3. Try smart color keying (studio shots)
    if result_img is None and engine in ('auto', 'color_key'):
        result_img, method_used = remove_bg_color_smart(filepath)

    # 4. Last resort
    if result_img is None:
        try:
            import cv2
            result_img, method_used = remove_bg_car_opencv(filepath, 'draft')
        except Exception:
            pass

    if result_img is None:
        raise RuntimeError("All background removal engines failed. Check if OpenCV is installed: pip install opencv-python")

    # Optional despill
    if despill_enable and result_img is not None:
        result_img = despill(result_img)

    return result_img, method_used


def generate_mask_preview(nobg_img, width=300, height=200):
    """
    Generate a checkerboard preview of the cutout result.
    Returns base64 PNG string.
    """
    checker = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(checker)
    tile = 12
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            color = (200, 200, 200) if (x // tile + y // tile) % 2 == 0 else (240, 240, 240)
            draw.rectangle([x, y, x+tile, y+tile], fill=color)

    preview = nobg_img.copy()
    preview.thumbnail((width, height), Image.LANCZOS)
    px = (width - preview.width) // 2
    py = (height - preview.height) // 2
    checker.paste(preview, (px, py), preview)

    buf = io.BytesIO()
    checker.save(buf, 'PNG')
    return base64.b64encode(buf.getvalue()).decode()


# ═══════════════════════════════════════════════════════════
#  IMAGE PROCESSING MODULE
# ═══════════════════════════════════════════════════════════

def apply_background(fg_img, bg_color, width=1200, height=800, lighting=1.0, shadow=True,
                     shadow_intensity=0.4, shadow_blur=15, position_y_offset=0.05):
    canvas = Image.new('RGB', (width, height), bg_color)
    fg = fg_img.copy()
    max_w, max_h = int(width * 0.85), int(height * 0.75)
    fw, fh = fg.size
    scale = min(max_w / fw, max_h / fh, 1.0)
    fw, fh = int(fw * scale), int(fh * scale)
    fg = fg.resize((fw, fh), Image.LANCZOS)

    if lighting != 1.0:
        enhancer = ImageEnhance.Brightness(fg)
        fg = enhancer.enhance(lighting)

    x = (width - fw) // 2
    y = (height - fh) // 2 + int(height * position_y_offset)

    if shadow:
        shadow_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shadow_layer)
        sh = max(int(fh * 0.08), 1)
        alpha_val = int(255 * shadow_intensity)
        draw.ellipse(
            [x + fw // 6, y + fh - sh, x + fw - fw // 6, y + fh + sh],
            fill=(0, 0, 0, alpha_val)
        )
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(shadow_blur))
        canvas = Image.alpha_composite(canvas.convert('RGBA'), shadow_layer).convert('RGB')

    canvas.paste(fg, (x, y), fg if fg.mode == 'RGBA' else None)
    return canvas


def add_watermark(img):
    img = img.copy().convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    size = max(40, img.width // 12)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), "SAMPLE", font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((img.width - tw) // 2, (img.height - th) // 2), "SAMPLE", fill=(255, 255, 255, 38), font=font)
    return Image.alpha_composite(img, overlay).convert('RGB')


# ═══════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html', backgrounds=BACKGROUNDS, swatches=SWATCHES)

@app.route('/api/upload', methods=['POST'])
def upload():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files received'}), 400
    results = []
    errors = []
    for f in files:
        if not f or f.filename == '':
            continue
        try:
            uid = str(uuid.uuid4())[:12]
            original_name = f.filename or 'car.jpg'
            ext = os.path.splitext(original_name)[1].lower()
            if ext not in ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff'):
                ext = '.jpg'
            fname = f"car_{uid}{ext}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            f.save(path)
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                errors.append(f'{original_name}: save failed')
                continue
            car_name = os.path.splitext(original_name)[0]
            car = CarImage(id=uid, filename=fname, car_name=car_name, original_path=path, status='uploaded')
            db.session.add(car)
            results.append({'id': uid, 'filename': fname, 'car_name': car_name, 'original_url': '/' + path})
        except Exception as e:
            errors.append(str(e))
    if not results and errors:
        db.session.rollback()
        return jsonify({'error': '; '.join(errors)}), 500
    db.session.commit()
    return jsonify(results)


@app.route('/api/remove-bg/<image_id>', methods=['POST'])
def remove_bg_route(image_id):
    """
    Enhanced background removal with engine/quality selection.
    Body params:
      engine:  auto | rembg | grabcut | color_key
      quality: draft | standard | high | ultra
      despill: bool
    """
    car = CarImage.query.get_or_404(image_id)
    data = request.get_json(silent=True) or {}

    engine  = data.get('engine', 'auto')
    quality = data.get('quality', 'standard')
    despill_enable = bool(data.get('despill', False))

    result, method_used = remove_bg_ai(
        car.original_path,
        quality=quality,
        engine=engine,
        despill_enable=despill_enable
    )

    out_name = f"nobg_{car.id}.png"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_name)
    result.save(out_path, 'PNG')

    # Generate checkerboard preview thumbnail
    preview_b64 = generate_mask_preview(result)

    car.nobg_path = out_path
    car.status = 'bg_removed'
    car.bg_removal_method = method_used
    car.bg_removal_quality = quality
    db.session.add(CreditLog(action=f'BG Removal [{method_used}]', cost=0))
    db.session.commit()

    return jsonify({
        'nobg_url': '/' + out_path,
        'status': 'bg_removed',
        'method': method_used,
        'quality': quality,
        'preview_b64': preview_b64,
        'image_size': list(result.size),
    })


@app.route('/api/remove-bg-batch', methods=['POST'])
def remove_bg_batch():
    """
    Batch background removal for multiple images.
    """
    data = request.get_json(silent=True) or {}
    ids     = data.get('ids', [])
    engine  = data.get('engine', 'auto')
    quality = data.get('quality', 'standard')
    despill = bool(data.get('despill', False))

    results = []
    for cid in ids:
        car = CarImage.query.get(cid)
        if not car:
            results.append({'id': cid, 'error': 'not found'})
            continue
        try:
            img, method = remove_bg_ai(car.original_path, quality=quality, engine=engine, despill_enable=despill)
            out_name = f"nobg_{car.id}.png"
            out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_name)
            img.save(out_path, 'PNG')
            car.nobg_path = out_path
            car.status = 'bg_removed'
            car.bg_removal_method = method
            car.bg_removal_quality = quality
            db.session.add(CreditLog(action=f'Batch BG Removal [{method}]', cost=0))
            results.append({'id': cid, 'nobg_url': '/' + out_path, 'method': method})
        except Exception as e:
            results.append({'id': cid, 'error': str(e)})

    db.session.commit()
    return jsonify({'results': results, 'done': len([r for r in results if 'error' not in r])})


@app.route('/api/refine-mask/<image_id>', methods=['POST'])
def refine_mask_route(image_id):
    """
    Re-refine an existing mask with edge smoothing.
    """
    car = CarImage.query.get_or_404(image_id)
    if not car.nobg_path or not os.path.exists(car.nobg_path):
        return jsonify({'error': 'No mask to refine'}), 400

    data = request.get_json(silent=True) or {}
    radius = int(data.get('radius', 2))

    img = Image.open(car.nobg_path).convert('RGBA')
    refined = refine_edges(img, radius=radius)
    refined.save(car.nobg_path, 'PNG')

    preview_b64 = generate_mask_preview(refined)
    db.session.add(CreditLog(action='Mask Refinement', cost=0))
    db.session.commit()

    return jsonify({
        'nobg_url': '/' + car.nobg_path,
        'preview_b64': preview_b64,
        'status': 'refined'
    })


@app.route('/api/apply-bg/<image_id>', methods=['POST'])
def apply_bg_route(image_id):
    car = CarImage.query.get_or_404(image_id)
    data = request.json or {}
    bg_id = data.get('bg_id', 'studio_white')
    custom_color = data.get('custom_color')
    lighting = float(data.get('lighting', 1.0))
    shadow = data.get('shadow', True)
    shadow_intensity = float(data.get('shadow_intensity', 0.4))
    shadow_blur = int(data.get('shadow_blur', 15))
    width = int(data.get('width', 1200))
    height = int(data.get('height', 800))

    if bg_id == 'custom' and custom_color:
        bg_rgb = hex_to_rgb(custom_color)
    else:
        all_bgs = BACKGROUNDS['studio'] + BACKGROUNDS['outdoor']
        found = next((b for b in all_bgs if b['id'] == bg_id), None)
        bg_rgb = hex_to_rgb(found['color']) if found else (255, 255, 255)

    src_path = car.nobg_path or car.original_path
    fg = Image.open(src_path).convert('RGBA')
    result = apply_background(fg, bg_rgb, width, height, lighting, shadow, shadow_intensity, shadow_blur)
    out_name = f"proc_{car.id}.jpg"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_name)
    result.save(out_path, 'JPEG', quality=92)
    car.processed_path = out_path
    car.status = 'completed'
    db.session.add(CreditLog(action='Apply Background', cost=0))
    db.session.commit()
    return jsonify({'processed_url': '/' + out_path, 'status': 'completed'})


@app.route('/api/watermark/<image_id>', methods=['POST'])
def watermark_route(image_id):
    car = CarImage.query.get_or_404(image_id)
    src = car.processed_path or car.nobg_path or car.original_path
    img = Image.open(src).convert('RGBA')
    result = add_watermark(img)
    out_name = f"wm_{car.id}.jpg"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_name)
    result.save(out_path, 'JPEG', quality=88)
    car.processed_path = out_path
    db.session.commit()
    return jsonify({'processed_url': '/' + out_path})


@app.route('/api/resize/<image_id>', methods=['POST'])
def resize_route(image_id):
    car = CarImage.query.get_or_404(image_id)
    data = request.json or {}
    w = int(data.get('width', 1200))
    h = int(data.get('height', 800))
    src = car.processed_path or car.nobg_path or car.original_path
    img = Image.open(src).convert('RGB')
    img = img.resize((w, h), Image.LANCZOS)
    out_name = f"rsz_{car.id}.jpg"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_name)
    img.save(out_path, 'JPEG', quality=92)
    car.processed_path = out_path
    db.session.commit()
    return jsonify({'processed_url': '/' + out_path, 'width': w, 'height': h})


@app.route('/api/direct-upload', methods=['POST'])
def direct_upload():
    ids = request.json.get('ids', [])
    for cid in ids:
        car = CarImage.query.get(cid)
        if car:
            car.in_gallery = True
    db.session.commit()
    return jsonify({'ok': True})


@app.route('/api/apply-to-all', methods=['POST'])
def apply_to_all():
    data = request.json or {}
    ids = data.get('ids', [])
    bg_id = data.get('bg_id', 'studio_white')
    custom_color = data.get('custom_color')
    lighting = float(data.get('lighting', 1.0))
    shadow = data.get('shadow', True)
    width = int(data.get('width', 1200))
    height = int(data.get('height', 800))
    engine  = data.get('engine', 'auto')
    quality = data.get('quality', 'standard')

    if bg_id == 'custom' and custom_color:
        bg_rgb = hex_to_rgb(custom_color)
    else:
        all_bgs = BACKGROUNDS['studio'] + BACKGROUNDS['outdoor']
        found = next((b for b in all_bgs if b['id'] == bg_id), None)
        bg_rgb = hex_to_rgb(found['color']) if found else (255, 255, 255)

    done = 0
    for cid in ids:
        car = CarImage.query.get(cid)
        if not car: continue
        if not car.nobg_path:
            result, method = remove_bg_ai(car.original_path, quality=quality, engine=engine)
            out = os.path.join(app.config['PROCESSED_FOLDER'], f"nobg_{car.id}.png")
            result.save(out, 'PNG')
            car.nobg_path = out
            car.bg_removal_method = method
        fg = Image.open(car.nobg_path).convert('RGBA')
        proc = apply_background(fg, bg_rgb, width, height, lighting, shadow)
        out = os.path.join(app.config['PROCESSED_FOLDER'], f"proc_{car.id}.jpg")
        proc.save(out, 'JPEG', quality=92)
        car.processed_path = out
        car.status = 'completed'
        db.session.add(CreditLog(action='Bulk Process', cost=0))
        done += 1
    db.session.commit()
    return jsonify({'done': done, 'total': len(ids)})


@app.route('/api/save-gallery', methods=['POST'])
def save_gallery():
    ids = request.json.get('ids', [])
    for cid in ids:
        car = CarImage.query.get(cid)
        if car:
            car.in_gallery = True
    db.session.commit()
    return jsonify({'ok': True})


@app.route('/api/gallery')
def get_gallery():
    cars = CarImage.query.filter_by(in_gallery=True).order_by(CarImage.created_at.desc()).all()
    return jsonify([{
        'id': c.id, 'car_name': c.car_name, 'status': c.status,
        'original_url': '/' + c.original_path if c.original_path else None,
        'nobg_url': '/' + c.nobg_path if c.nobg_path else None,
        'processed_url': '/' + c.processed_path if c.processed_path else None,
        'bg_removal_method': c.bg_removal_method,
        'bg_removal_quality': c.bg_removal_quality,
    } for c in cars])


@app.route('/api/gallery/<image_id>', methods=['DELETE'])
def delete_gallery(image_id):
    car = CarImage.query.get_or_404(image_id)
    for p in [car.original_path, car.nobg_path, car.processed_path]:
        if p and os.path.exists(p): os.remove(p)
    db.session.delete(car)
    db.session.commit()
    return jsonify({'ok': True})


@app.route('/api/credit-logs')
def credit_logs():
    logs = CreditLog.query.order_by(CreditLog.timestamp.desc()).limit(100).all()
    return jsonify([{'action': l.action, 'cost': l.cost, 'time': l.timestamp.isoformat()} for l in logs])


@app.route('/api/download/<image_id>')
def download_image(image_id):
    car = CarImage.query.get_or_404(image_id)
    path = car.processed_path or car.original_path
    return send_file(path, as_attachment=True, download_name=f"car_{car.car_name}.jpg")


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5055)