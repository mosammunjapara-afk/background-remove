"""
AutoLens Pro — Setup Script
Run this ONCE before starting the app for the first time:
    python setup.py
Then start the app:
    python app.py
"""
import os, sys

REQUIRED_DIRS = [
    'static',
    'static/uploads',
    'static/processed',
    'templates',
]

def setup():
    print("=" * 50)
    print("  AutoLens Pro — Setup")
    print("=" * 50)

    # 1. Create required directories
    for d in REQUIRED_DIRS:
        os.makedirs(d, exist_ok=True)
        print(f"  ✅ Directory: {d}")

    # 2. Move index.html to templates/ if needed
    if os.path.exists('index.html') and not os.path.exists('templates/index.html'):
        import shutil
        shutil.copy('index.html', 'templates/index.html')
        print("  ✅ Copied index.html → templates/index.html")
    elif os.path.exists('templates/index.html'):
        print("  ✅ templates/index.html already exists")
    else:
        print("  ⚠️  WARNING: index.html not found! Copy it to templates/index.html")

    # 3. Check required packages
    print("\n  Checking packages...")
    packages = {
        'flask': 'Flask',
        'flask_sqlalchemy': 'Flask-SQLAlchemy',
        'PIL': 'Pillow',
    }
    optional = {
        'rembg': 'rembg (AI background removal — recommended)',
        'cv2': 'opencv-python (GrabCut engine fallback)',
        'numpy': 'numpy',
    }
    for mod, name in packages.items():
        try:
            __import__(mod)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ MISSING: {name}  →  pip install {name.lower()}")

    print("\n  Optional packages:")
    for mod, name in optional.items():
        try:
            __import__(mod)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️  Not installed: {name}")

    # 4. Initialize DB
    print("\n  Initializing database...")
    try:
        from app import app, db
        with app.app_context():
            db.create_all()
        print("  ✅ Database ready (autolens.db)")
    except Exception as e:
        print(f"  ⚠️  DB init skipped (will auto-init on first run): {e}")

    print("\n" + "=" * 50)
    print("  Setup complete! Run:  python app.py")
    print("  Then open:  http://localhost:5055")
    print("=" * 50)

if __name__ == '__main__':
    setup()