
# AI Virtual Ornament Try-On System

A real-time desktop application that allows users to virtually try on **earrings, necklaces, or both** using a webcam feed.  
The system provides an interactive jewelry preview experience with dynamic alignment, scaling, and design switching.

---

##  Features
- Real-time webcam-based ornament try-on
- Supports **Earrings / Necklace / Both**
- Dynamic alignment with face movement
- Adaptive scaling based on face size
- Transparent PNG overlay rendering
- Next / Previous ornament switching
- User preference selection
- Lightweight desktop prototype

---

## Project Structure
AI_Virtual_Ornament_TryOn/
 - main.py
 - README.md
 - ornaments/
     - earring
        - earring1.png
        - earring2.png
        - ...
    - necklace
       - necklace1.png
       - necklace2.png
       - ...
  - models
      - shape_predictor_68_face_landmarks.dat
      - deploy.prototxt
      - res10_300x300_ssd_iter_140000.caffemodel

# Technologies Used
 - Python
 - OpenCV
 - NumPy
 - dlib
 - Deep Learning-based Face Detection
 - Facial Landmark Localization
 - PNG Alpha Blending
 - Real-time Computer Vision

# How to Run
 1) Install dependencies
   - pip install opencv-python numpy dlib
 2) Run the project
 - python main.py

# Controls
- E → Next Earring
- Q → Previous Earring
- N → Next Necklace
- B → Previous Necklace
- ESC → Exit

# Applications
- Virtual jewelry shopping
- Smart retail mirrors
- E-commerce product preview
- AI showroom experience
- Personalized ornament recommendation systems

# Future Enhancements
- 3D jewelry warping
- AR smart mirror
- Mobile app deployment
- Personalized jewelry suggestions
- E-commerce website integratio
