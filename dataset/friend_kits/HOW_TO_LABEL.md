# How to label (for friends)

1. Install Python 3.10+ and the packages:
   `pip install pillow opencv-python` (numpy comes with them)
2. Open a terminal in the **outfit-ai** folder.
3. Prepare the top dataset folder (no friend name is needed):

   ```
   python 1_label/make_kit.py
   ```

4. Put the pictures to label in `dataset/top/images`, then run:

   ```
   python 1_label/annotate.py --garment top
   ```

5. For each picture:
   - Choose **Garment**, select length, type, and pattern together, draw one
     box around the garment, then press **Enter** to save that region.
   - For tops, choose **Neck** to select a neck type and draw its own box.
     Choose **Sleeves**, select the sleeve type, draw its box, and press
       **Enter**. The box's horizontal position automatically determines
       image-left or image-right; add the other sleeve as a second region.
       Neck and sleeve types are selected as labels, not inferred from the
       box location.
   - The garment colour is detected automatically when garment labels are
     saved.
   - **Left / Right** arrows move between pictures, **Ctrl+N** jumps to the
     next unlabelled picture, and **Del** removes a bad picture.
6. Everything saves automatically into `dataset/top/annotations.json`. Send
   the `dataset/top` folder back when done.

Tips
- Look at `LABELS.md` inside the kit for the exact label words.
- Unsure about a category? Leave it empty (skip it) — the AI can handle
  missing labels.
- One garment per picture, front view, plain background = best results.
