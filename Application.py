import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import shutil

# Configuration de la page
st.set_page_config(
    page_title="Détection de Poubelles",
    page_icon="🗑️",
    layout="wide"
)

# Titre de l'application
st.title("🗑️ Détection de Poubelles - Pleine ou Vide")
st.markdown("---")


# Charger le modèle YOLO
@st.cache_resource
def load_model():
    try:
        # Vérifier d'abord si un modèle a été uploadé
        if os.path.exists("uploaded_model.pt"):
            model_path = "uploaded_model.pt"
            st.sidebar.info("🔄 Chargement du modèle uploadé...")
        else:
            # Sinon utiliser le modèle par défaut
            model_path = 'runs_training/yolov8_training2/weights/best.pt'
            st.sidebar.info("🔄 Chargement du modèle par défaut...")
        
        # Charger le modèle
        model = YOLO(model_path)
        st.sidebar.success("✅ Modèle chargé avec succès!")
        
        # Afficher les informations du modèle
        st.sidebar.subheader("📋 Classes du modèle")
        for class_id, class_name in model.names.items():
            st.sidebar.write(f"- Classe {class_id}: {class_name}")
        
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Erreur lors du chargement: {e}")
        return None

# Fonction de prédiction SIMPLIFIÉE
def predict_image(model, image, confidence_threshold):
    try:
        # Effectuer la prédiction
        results = model(image, conf=confidence_threshold, verbose=False)
        
        detections = []
        image_with_boxes = image.copy()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    # Récupérer les informations de la boîte
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Obtenir le nom de la classe depuis le modèle
                    class_name = model.names.get(cls, f"Classe_{cls}")
                    
                    # AFFICHER DIRECTEMENT CE QUE LE MODÈLE DIT
                    if "plein" in class_name.lower() or "pleine" in class_name.lower():
                        color = (0, 255, 0)  # Vert pour pleine
                        display_label = "PLEINE"
                    elif "vide" in class_name.lower():
                        color = (255, 0, 0)  # Rouge pour vide
                        display_label = "VIDE"
                    else:
                        # Si le nom ne contient pas "plein" ou "vide", utiliser le nom tel quel
                        color = (0, 255, 255)  # Jaune pour autres
                        display_label = class_name.upper()
                    
                    # Dessiner la boîte sur l'image
                    cv2.rectangle(image_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    
                    # Ajouter l'étiquette
                    label_text = f"{display_label} {conf:.2f}"
                    cv2.putText(image_with_boxes, label_text, (int(x1), int(y1)-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Stocker les détections
                    detections.append({
                        'class': cls,
                        'class_name': class_name,
                        'confidence': conf,
                        'label_display': display_label,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
        
        return image_with_boxes, detections
        
    except Exception as e:
        st.sidebar.error(f"❌ Erreur lors de la prédiction: {e}")
        return image, []

# Sidebar pour les paramètres
st.sidebar.title("⚙️ Paramètres")

# Affichage du statut du modèle
st.sidebar.subheader("📊 Statut du modèle")
if os.path.exists("uploaded_model.pt"):
    st.sidebar.success("✅ Modèle personnalisé chargé")
elif os.path.exists('runs_training/yolov8_training2/weights/best.pt'):
    st.sidebar.info("ℹ️ Modèle par défaut chargé")
else:
    st.sidebar.error("❌ Aucun modèle trouvé")

# Paramètres de détection
st.sidebar.markdown("---")
confidence_threshold = st.sidebar.slider("Seuil de confiance", 0.6, 0.8, 0.5, 0.01)

# Charger le modèle
model = load_model()

# Section upload du modèle À LA FIN
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Gestion des modèles")

# Bouton pour uploader un modèle
with open("runs_training/yolov8_training2/weights/best.pt", "rb") as f:
        st.sidebar.download_button("⬇️ Télécharger le modèle", f, "best.pt")


# Bouton pour supprimer le modèle uploadé
if os.path.exists("uploaded_model.pt"):
    if st.sidebar.button("🗑️ Supprimer le modèle uploadé", type="secondary", use_container_width=True):
        try:
            os.remove("uploaded_model.pt")
            st.sidebar.success("✅ Modèle uploadé supprimé")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Erreur lors de la suppression: {e}")

# Section principale
col1, col2 = st.columns(2)

with col1:
    st.header("📸 Upload d'Image")
    
    # Upload d'image
    uploaded_file = st.file_uploader(
        "Choisissez une image de poubelle",
        type=['jpg', 'jpeg', 'png'],
        help="Uploader une image contenant une poubelle"
    )
    
    if uploaded_file is not None:
        # Afficher l'image originale
        image = Image.open(uploaded_file)
        st.image(image, caption="Image originale", use_column_width=True)
        
        # Convertir en format OpenCV
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

with col2:
    st.header("🔍 Résultats de la Détection")
    
    if uploaded_file is not None and model is not None:
        # Bouton de prédiction
        if st.button("🔍 Analyser l'image", type="primary", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                # Faire la prédiction
                image_with_boxes, detections = predict_image(model, image_cv.copy(), confidence_threshold)
                
                # Convertir pour l'affichage
                image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
                
                # Afficher l'image avec les détections
                st.image(image_with_boxes_rgb, caption="Image avec détections", use_column_width=True)
                
                # Afficher les résultats
                if detections:
                    st.subheader("📊 Résultats de détection")
                    
                    # Afficher chaque détection
                    for i, det in enumerate(detections):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if det['label_display'] == "PLEINE":
                                st.success("🗑️ PLEINE")
                            elif det['label_display'] == "VIDE":
                                st.info("🗑️ VIDE")
                            else:
                                st.warning(f"🗑️ {det['label_display']}")
                        
                        with col2:
                            st.write(f"Confiance: **{det['confidence']:.3f}**")
                            st.write(f"Classe: {det['class_name']} (ID: {det['class']})")
                    
                    # CONCLUSION - AFFICHER SI LA POUBELLE EST PLEINE OU VIDE
                    st.subheader("🎯 CONCLUSION FINALE")
                    
                    pleine_count = len([d for d in detections if d['label_display'] == "PLEINE"])
                    vide_count = len([d for d in detections if d['label_display'] == "VIDE"])
                    
                    if pleine_count > 0 and vide_count == 0:
                        st.success("## ✅ LA POUBELLE EST PLEINE")
                    elif vide_count > 0 and pleine_count == 0:
                        st.info("## ❌ LA POUBELLE EST VIDE")
                    elif pleine_count > 0 and vide_count > 0:
                        st.warning(f"## 🤔 RÉSULTAT MIXTE - {pleine_count} poubelle(s) pleine(s) et {vide_count} poubelle(s) vide(s)")
                    else:
                        st.warning("## 🔍 AUTRES OBJETS DÉTECTÉS")
                        
                else:
                    st.error("❌ Aucune poubelle détectée dans l'image")
                    st.info("""
                    **Suggestions :**
                    - 📉 Baissez le seuil de confiance
                    - 📸 Utilisez une image plus nette
                    - 🎯 Assurez-vous que la poubelle est bien visible
                    """)

    elif model is None:
        st.error("❌ Modèle non chargé - Veuillez uploader un modèle ou vérifier le chemin par défaut")

# Section d'information
st.markdown("---")
st.subheader("ℹ️ Comment ça marche ?")

st.markdown("""
**📁 Upload de modèle :**
1. Allez dans la section "Gestion des modèles" en bas de la sidebar
2. Cliquez sur "Choisissez votre fichier de modèle"
3. Sélectionnez votre fichier .pt entraîné
4. Le modèle sera automatiquement chargé

**🎯 Détection d'images :**
1. 📸 **Uploader** une image de poubelle
2. ⚙️ **Ajuster** le seuil de confiance si nécessaire
3. 🔍 **Cliquer** sur "Analyser l'image"
4. 📊 **Vérifier** les résultats et la conclusion

**Légende des couleurs :**
- 🟢 **VERT** : Poubelle **PLEINE**
- 🔴 **ROUGE** : Poubelle **VIDE**
- 🟡 **JAUNE** : Autre type de détection
""")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Application de détection de poubelles - Support modèles personnalisés"
    "</div>",
    unsafe_allow_html=True
)