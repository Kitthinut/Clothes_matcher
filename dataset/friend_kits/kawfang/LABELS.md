# BOTTOM labels for 'kawfang'

- Waist length (pick ONE): low-rise / mid-rise / high-rise
- Length (pick ONE): mini / knee-length / cropped / full-length / shorts
- Pattern (tick ALL that apply, or none): solid / dotted / striped / checked / floral / graphic
- Fit type (pick ONE): skinny / straight / wide-leg / relaxed / flare

How to label:
1. python -m pip install pillow opencv-python   (once)
2. python 1_label/annotate.py <this kit folder> --garment bottom
3. Add a Garment region with its labels, then add separate typed
   Neck and left/right Sleeve regions when those categories exist.
   Draw each region and press Enter to save it.
4. When done, send the whole folder back.