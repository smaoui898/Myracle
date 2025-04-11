# Myracle
Système de Vision pour une Canne Intelligente pour Malvoyants
**Myracle** est une solution d’assistance visuelle intelligente conçue pour améliorer la mobilité et la sécurité des personnes malvoyantes. Grâce à la détection d’obstacles en temps réel et un retour vocal, ce système rend la navigation plus sûre et intuitive.

#Objectifs du Projet
Notre solution vise à répondre aux besoins suivants :
- Détection et identification des obstacles en temps réel
- Retour vocal automatique à l’utilisateur
- Système portable et autonome (Raspberry Pi ou smartphone)
- Possibilité d’extension avec une interface mobile ou un module interactif vocal

#Technologies et Outils Utilisés

**OpenCV** :Capture et traitement d’image

**YOLOv8** : Détection en temps réel des objets/obstacles

**gTTS + sounddevice** : Synthèse et lecture vocale des alertes

**Streamlit (optionnel)** : Interface simple et interactive  (Possibilité)

#Fonctionnement du Système
1. **Capture vidéo** en temps réel via une caméra montée sur la canne.
2. **Détection d’obstacles** à l’aide d’un modèle de vision par ordinateur (YOLO).
3. **Identification** de l’objet (type, position : devant/gauche/droite).
4. **Génération d’une alerte vocale** avec gTTS (ex. : *Person ahead !"*).
5. (à étudier) **Interaction vocale possible** : ex. *"Quelle est la distance de l’obstacle ?"*

#Exemples d’Objets Détectés :

| Objet | Niveau de Priorité | Alerte Vocale |

| Personne | Haute | *"Person approaching!"* |

| Porte | Haute | *"Door close ahead!"* |

| Escaliers | Haute | *"Stairs warning!"* |

| Voiture / Chien | Haute | *"Vehicle approaching!" / "Dog close by!"* | 

| Chaise / Lit / Tasse | Moyenne | *"Chair ahead!" / "Cup to your left!"* |

| Téléphone | faible | *"Mobile phone ahead!"* |

 #Réalisé par 
 Ahmed Smaoui/Donia Bahloul/Ala Belguith : Étudiants en Génie Informatique à l’ENIS
 
Projet encadré dans le cadre du module de Computer Vision






