import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict

# Exemple de données
data = {
    'keywords': [
        # Originaux + quelques variations
        "chat dormir canapé",
        "chien courir parc",
        "enfant jouer jardin",
        "oiseau chanter matin",
        "livre lire soirée",
        "chat manger poisson",
        "soleil briller ciel bleu",
        "pluie tomber fenêtre",
        "voiture rouge garer rue",
        "fleurs éclore printemps jardin",
        "musique douce jouer radio",
        "montagnes enneigées lever soleil",
        "chocolat fondre langue délicieux",
        "vagues océan murmurer plage",
        "étoiles scintiller nuit étoilée",
        "feuilles automne tomber sol",
        "oiseaux gazouiller arbres matin",
        "enfants rire jouer parc",
        "nuages flotter ciel calme",
        "livre ouvert table raconter histoire",
        "lumière douce éclairer pièce",
        "fleur rose éclore jardin",
        "chaton jouer pelote laine",
        "café chaud tasse matin",
        "vague mer toucher pieds",
        "chien aboyer porte",
        "vent souffler feuilles automne",
        "lune briller ciel nocturne",
        "enfants jouer ballon parc",
        "livre passionnant lire soir",
        "fleurs colorées jardin printemps",
        "musique classique résonner salle",
        "vagues déferler plage sable",
        "oiseaux migrer ciel automne",
        "montagnes majestueuses lever soleil",
        "café chaud réconfortant matin",
        "chat ronronner canapé douillet",
        "étoiles filantes nuit étoilée",
        "feuilles rouges tomber automne",
        "enfants rire jouer cache-cache",
        "nuages cotonneux flotter ciel",
        "livre ancien raconter histoire",
        "lumière tamisée éclairer pièce",
        "fleur délicate parfumer jardin",
        "chaton curieux explorer maison",
        "café aromatique réveiller matin",
        "vague douce caresser pieds",
        "bateau naviguer mer calme",
        "papillon voleter fleur parfumée",
        "cheval galoper prairie verte",
        "artiste peindre toile colorée",
        "cuisinier préparer repas délicieux",
        "cycliste pédaler route campagne",
        "photographe capturer moment magique",
        "musicien jouer mélodie envoûtante",
        "écrivain écrire roman captivant",
        "jardinier planter graines potager",
        "randonneur gravir sommet montagne",
        "plongeur explorer fond océan",
        "astronome observer étoiles télescope",
        "architecte dessiner plans bâtiment",
        "sculpteur façonner statue argile",
        "danseur interpréter ballet gracieux",
        "pâtissier décorer gâteau anniversaire",
        "viticulteur récolter raisins vigne",
        "surfeur glisser vague géante",
        "alpiniste escalader paroi rocheuse",
        "pêcheur lancer ligne rivière tranquille",
        "chanteur interpréter chanson émouvante",
        "professeur enseigner cours passionnant",
        "médecin ausculter patient cabinet",
        "ingénieur concevoir projet innovant",
        "journaliste rédiger article actualité",
        "pilote décoller avion piste",
        "chercheur analyser données laboratoire",
        "boulanger cuire pain four chaud",
        "étudiant réviser examens bibliothèque",
        "couple danser mariage romantique",
        "enfant souffler bougies gâteau anniversaire",
        "famille pique-niquer parc ensoleillé",
        "amis discuter café terrasse",
        "artiste exposer œuvres galerie",
        "écrivain dédicacer livre librairie",
        "musiciens répéter concert studio",
        "athlète courir marathon ville",
        "acteur répéter scène théâtre",
        "réalisateur tourner film plateau",
        "chef cuisiner plat restaurant",
        "voyageur explorer ville inconnue",
        "guide raconter histoire musée",
        "pilote atterrir avion aéroport",
        "chercheur découvrir remède laboratoire",
        "chat siamois miauler fort",          # Plus spécifique
        "chien berger allemand courir champs", # Plus spécifique
        "enfant dessiner craie trottoir",     # Action différente
        "abeille butiner fleur lavande",      # Plus spécifique et action différente

        # Nouveaux Ajouts - Nature et Météo
        "brouillard épais recouvrir vallée",
        "rosée matin perler herbe",
        "arc-en-ciel apparaître après pluie",
        "orage éclater ciel sombre",
        "neige tomber doucement paysage",
        "vent violent secouer arbres",
        "grêle frapper toit maison",
        "soleil coucher derrière horizon",
        "brise légère rafraîchir air",
        "feu crépiter cheminée salon",
        "cascade rugir montagne rocheuse",
        "ruisseau serpenter forêt calme",
        "lac refléter montagnes ciel",
        "éclair zébrer ciel nuit",            # Ajout Météo
        "tsunami dévaster côte",              # Ajout Catastrophe naturelle
        "éruption volcanique projeter cendres", # Ajout Catastrophe naturelle
        "aurore boréale danser ciel polaire", # Ajout Phénomène naturel

        # Nouveaux Ajouts - Activités et Objets
        "ordinateur portable afficher écran bleu",
        "téléphone portable sonner sac",
        "clé usb contenir données importantes",
        "stylo bille écrire papier blanc",
        "fenêtre ouverte laisser entrer air frais",
        "porte claquer bruit vent",
        "rideaux flotter fenêtre ouverte",
        "miroir refléter lumière pièce",
        "horloge murale indiquer heure",
        "tapis persan décorer sol salon",
        "vase cristal contenir fleurs fraîches",
        "bougie parfumée brûler table basse",
        "théière siffler cuisine",
        "pizza chaude sortir four",
        "verre vin rouge table salle manger",
        "télévision diffuser match football",  # Ajout Objet/Activité
        "réfrigérateur conserver aliments frais", # Ajout Objet
        "machine à laver nettoyer vêtements",  # Ajout Objet/Activité
        "aspirateur aspirer poussière tapis", # Ajout Objet/Activité
        "livre recette expliquer plat",        # Ajout Objet/Activité

        # Nouveaux Ajouts - Émotions et Sensations
        "joie intense illuminer visage",
        "tristesse profonde envahir cœur",
        "colère sourde gronder ventre",
        "peur panique glacer sang",
        "surprise agréable écarquiller yeux",
        "calme serein apaiser esprit",
        "excitation vibrante parcourir corps",
        "douleur aiguë lancer bras",
        "fatigue intense peser paupières",
        "odeur agréable chatouiller narines",
        "goût amer persister bouche",
        "son strident agresser oreilles",
        "toucher doux réconforter peau",
        "silence pesant envahir pièce",
        "nostalgie envahir souvenir passé",    # Ajout Émotion
        "curiosité piquer intérêt enfant",    # Ajout Émotion/État
        "fierté gonfler poitrine réussite",   # Ajout Émotion
        "honte rougir joues erreur",         # Ajout Émotion
        "frustration monter échec",          # Ajout Émotion

        #Nouveaux ajouts - Actions plus complexes
        "étudiant surligner passage important livre",
        "programmeur déboguer code informatique",
        "scientifique mélanger produits chimiques laboratoire",
        "ouvrier construire mur briques",
        "conducteur bus conduire passagers ville",
        "infirmière administrer piqûre patient",
        "pompier éteindre incendie bâtiment",
        "policier diriger circulation carrefour",
        "serveur prendre commande restaurant",
        "caissier scanner articles supermarché",
        "bibliothécaire ranger livres étagères",
        "comptable vérifier factures bureau",
        "traducteur traduire document langue étrangère",
        "designer créer logo entreprise",
        "développeur web coder site internet",
        "agriculteur labourer champ tracteur",   # Ajout Action/Profession
        "mécanicien réparer moteur voiture",    # Ajout Action/Profession
        "électricien installer câblage maison",  # Ajout Action/Profession
        "plombier réparer fuite eau cuisine",  # Ajout Action/Profession
        "juge rendre verdict tribunal",       # Ajout Action/Profession

        # Ajouts avec pronoms personnels (simples)
        "je marcher parc",
        "tu lire livre",
        "il jouer football",
        "elle chanter chanson",
        "nous manger pizza",
        "vous regarder film",
        "ils parler amis",
        "elles danser fête",
        "on aller cinéma",  # Pronom "on"

        # Ajouts avec pronoms personnels (un peu plus complexes)
        "je penser vacances",
        "tu espérer beau temps",
        "il croire réussite",
        "elle vouloir voyager",
        "nous devoir travailler",
        "vous pouvoir aider",
        "ils savoir réponse",
        "elles aimer musique",
        "on devoir partir",

        # Ajouts avec pronoms personnels (complexes, avec propositions subordonnées, etc.)
        "je croire que tu réussir",
        "tu savoir où il aller",
        "il penser ce qu' elle dire",
        "elle espérer que nous venir",
        "nous vouloir que vous comprendre",
        "vous savoir comment ils faire",
        "ils dire que elles arriver",
        "elles penser à ce que on dire",
        "on espérer que tout aller bien",
        "je me souvenir de toi",
        "tu te demander pourquoi",
        "il se préparer pour examen",
        "elle se maquiller avant sortir",
        "nous nous rencontrer café",
        "vous vous inquiéter pour rien",
        "ils se disputer souvent",
        "elles se promener parc",
        "on se demander ce qui se passer",

        # Pronoms relatifs et compléments
        "livre que je lire intéressant",
        "film que tu regarder hier",
        "personne à qui il parler",
        "chanson qu' elle chanter",
        "endroit où nous aller",
        "raison pour laquelle vous partir",
        "chose dont ils parler",
        "moment où elles arriver",
        "façon dont on faire",
        "fille avec qui je parler",
        "garçon dont tu te moquer",
        "problème auquel il penser",
        "solution qu' elle trouver",
        "ville d'où nous venir",
        "pays où vous vivre",
        "personnes avec qui ils travailler",
        "amis à qui elles écrire",
        "sujet sur lequel on discuter",

        #Pronoms démonstratifs
        "celui-ci être mon livre",
        "celle-là être ta voiture",
        "ceux-ci être mes amis",
        "celles-là être leurs affaires",
        "ceci être important",
        "cela être intéressant",
        "ceux que je préférer",
        "celles que tu choisir",

        #Pronoms possessifs
        "le mien être bleu",                # Phrase complète pour clarté
        "le tien être plus grand",          # Phrase complète pour clarté
        "le sien fonctionner bien",         # Phrase complète pour clarté
        "la sienne être rouge",             # Phrase complète pour clarté
        "le nôtre arriver demain",          # Phrase complète pour clarté
        "le vôtre sembler neuf",            # Phrase complète pour clarté
        "les leurs être sur la table",      # Phrase complète pour clarté
        "la mienne être en retard",         # Phrase complète pour clarté
        "les tiens être ici",               # Phrase complète pour clarté
        "les siennes être perdues",         # Phrase complète pour clarté

        # ============================================
        # == AJOUTS PRÉEXISTANTS - Temps Verbaux ==
        # ============================================
        # Passé Composé
        "chat manger poisson hier",
        "nous arriver en retard ce matin",
        "elle lire ce livre la semaine dernière",
        "ils finir leur travail",
        "j'ai déjà voir ce film",
        "tu être allé au magasin",

        # Imparfait
        "enfant jouer souvent jardin",
        "il faire beau quand nous sortir",
        "oiseaux chanter tous les matins",
        "je penser à toi hier soir",
        "vous habiter ici avant ?",

        # Futur Simple
        "nous visiter musée demain",
        "il pleuvoir plus tard",
        "tu recevoir une lettre bientôt",
        "elles partir en vacances cet été",
        "je finir ce projet la semaine prochaine",

        # Conditionnel Présent
        "je aimer voyager plus souvent",
        "il pouvoir aider si tu demander",
        "nous préférer rester à la maison",
        "serait-il possible de venir ?",
        "vous devoir être plus prudent",

        # Subjonctif Présent (souvent après 'que')
        "il faut que tu étudier plus",
        "je vouloir que vous être heureux",
        "bien que il être fatigué il travailler",
        "avant que nous partir dire au revoir",
        "pour que ils comprendre expliquer",

        # ============================================
        # == AJOUTS PRÉEXISTANTS - Négation ==
        # ============================================
        "chat ne pas dormir canapé",
        "je ne jamais voir cet oiseau",
        "il ne plus avoir faim",
        "nous ne rien comprendre",
        "vous ne parler à personne",
        "elle n'aime ni chocolat ni café",

        # ============================================
        # == AJOUTS PRÉEXISTANTS - Interrogation ==
        # ============================================
        "où chien courir ?",
        "est-ce que enfant jouer jardin ?",
        "pourquoi oiseau chanter matin ?",
        "quand vous lire ce livre ?",
        "comment il faire ça ?",
        "qui arriver en premier ?",
        "tu aimer ce film ?",

        # ============================================
        # == AJOUTS PRÉEXISTANTS - Conjonctions ==
        # ============================================
        "chat dormir et chien veiller",
        "il pleut mais nous sortir quand même",
        "tu es fatigué car tu travailler tard",
        "je reste ici parce que j'attendre ami",
        "appelle-moi quand tu arriver",
        "si il faire beau nous aller plage",
        "il lit pendant que elle cuisiner",
        "bien que livre être long il intéressant",
        "tu peux choisir fromage ou dessert",

        # ============================================
        # == AJOUTS PRÉEXISTANTS - Adverbes ==
        # ============================================
        "chien courir rapidement parc",
        "oiseau chanter magnifiquement",
        "nous arriverons bientôt",
        "il pleut beaucoup aujourd'hui",
        "parle plus doucement s'il te plaît",
        "elle travaille toujours ici",
        "l'enfant mange lentement sa soupe",

        # ============================================
        # == AJOUTS PRÉEXISTANTS - Voix Passive ==
        # ============================================
        "poisson être mangé par chat",
        "lettre être écrite par Marie hier",
        "maison être construite en 2020",
        "règles devoir être respectées par tous",

        # ============================================
        # == NOUVEAUX AJOUTS - Divers Complexité ==
        # ============================================
        "malgré pluie nous sortir",             # Préposition + nom
        "grâce à aide tu réussir",             # Locution prépositive
        "homme marcher lentement canne",       # Complément de moyen
        "voyager avion être rapide",           # Infinitif sujet
        "aimer lire être bonne chose",         # Infinitif sujet
        "voiture garée devant maison",         # Participe passé adjectival
        "fleurs plantées jardinier sentir bon",# Participe passé adjectival complexe
        "c'est important de comprendre",       # Structure impersonnelle + infinitif
        "il est facile de faire erreur",       # Structure impersonnelle + infinitif
        "donner manger chat affamé",           # Double complément d'objet
        "il me expliquer problème clairement", # Pronom COI + COD
        "elle lui offrir cadeau surprise",     # Pronom COI + COD
        "penser avenir rendre anxieux",        # Proposition infinitive
        "voir enfants jouer rendre heureux",   # Proposition infinitive

        # ============================================
        # == NOUVEAUX AJOUTS - Comparaison et Superlatif ==
        # ============================================
        "chat être plus rapide que tortue",    # Comparatif supériorité (adj)
        "chien courir aussi vite que cheval",  # Comparatif égalité (adv)
        "il travailler moins que collègue",    # Comparatif infériorité (verbe)
        "ce livre être le plus intéressant",   # Superlatif supériorité (adj)
        "elle parler le moins fort",          # Superlatif infériorité (adv)
        "c'est la meilleure solution",         # Superlatif irrégulier (bon)
        "il fait le pire choix",              # Superlatif irrégulier (mauvais)

        # ============================================
        # == NOUVEAUX AJOUTS - Discours Rapporté (simple) ==
        # ============================================
        "il dire que il être fatigué",        # Discours indirect (présent -> imparfait)
        "elle demander si nous venir",         # Discours indirect (question oui/non)
        "ils demander où nous aller",          # Discours indirect (question mot interrogatif)
        "je penser que il pleuvoir demain",   # Discours indirect (opinion futur)

        # ============================================
        # == NOUVEAUX AJOUTS - Vocabulaire Spécifique (Tech, Science, Art) ==
        # ============================================
        "algorithme trier données efficacement", # Tech
        "téléscope spatial observer galaxie lointaine", # Science/Astro
        "molécule ADN contenir information génétique", # Science/Bio
        "peintre utiliser palette couleurs vives", # Art
        "sculpteur tailler marbre statue",       # Art
        "compositeur écrire symphonie orchestre",# Musique
        "réseau neuronal apprendre image",     # Tech/IA
        "photosynthèse convertir lumière énergie",# Science/Bio
        "danseur exécuter chorégraphie complexe",# Art/Danse

    ],
    'sentence': [
        # Originaux + quelques variations
        "Le chat dort sur le canapé.",
        "Le chien court dans le parc.",
        "L'enfant joue dans le jardin.",
        "L'oiseau chante le matin.",
        "Je lis un livre en soirée.",
        "Le chat mange du poisson.",
        "Le soleil brille dans le ciel bleu.",
        "La pluie tombe contre la fenêtre.",
        "La voiture rouge est garée dans la rue.",
        "Les fleurs éclosent au printemps dans le jardin.",
        "De la musique douce joue à la radio.",
        "Les montagnes enneigées se dressent au lever du soleil.",
        "Le chocolat fond sur la langue, c'est délicieux.",
        "Les vagues de l'océan murmurent sur la plage.",
        "Les étoiles scintillent dans la nuit étoilée.",
        "Les feuilles d'automne tombent sur le sol.",
        "Les oiseaux gazouillent dans les arbres le matin.",
        "Les enfants rient en jouant au parc.",
        "Les nuages flottent dans le ciel calme.",
        "Le livre ouvert sur la table raconte une histoire.",
        "Une lumière douce éclaire la pièce.",
        "La fleur rose éclot dans le jardin.",
        "Le chaton joue avec une pelote de laine.",
        "Le café chaud fume dans la tasse le matin.",
        "La vague de la mer touche mes pieds.",
        "Le chien aboie devant la porte.",
        "Le vent souffle sur les feuilles d'automne.",
        "La lune brille dans le ciel nocturne.",
        "Les enfants jouent au ballon dans le parc.",
        "Je lis un livre passionnant ce soir.",
        "Les fleurs colorées égaient le jardin au printemps.",
        "La musique classique résonne dans la salle.",
        "Les vagues déferlent sur la plage de sable.",
        "Les oiseaux migrent dans le ciel d'automne.",
        "Les montagnes majestueuses se dressent au lever du soleil.",
        "Un café chaud est réconfortant le matin.",
        "Le chat ronronne sur le canapé douillet.",
        "Les étoiles filantes illuminent la nuit étoilée.",
        "Les feuilles rouges tombent en automne.",
        "Les enfants rient en jouant à cache-cache.",
        "Les nuages cotonneux flottent dans le ciel.",
        "Le livre ancien raconte une histoire fascinante.",
        "Une lumière tamisée éclaire doucement la pièce.",
        "La fleur délicate parfume le jardin.",
        "Le chaton curieux explore la maison.",
        "Le café aromatique me réveille le matin.",
        "La vague douce me caresse les pieds.",
        "Le bateau navigue sur une mer calme.",
        "Le papillon volete autour d'une fleur parfumée.",
        "Le cheval galope dans une prairie verte.",
        "L'artiste peint une toile colorée.",
        "Le cuisinier prépare un repas délicieux.",
        "Le cycliste pédale sur une route de campagne.",
        "Le photographe capture un moment magique.",
        "Le musicien joue une mélodie envoûtante.",
        "L'écrivain écrit un roman captivant.",
        "Le jardinier plante des graines dans le potager.",
        "Le randonneur gravit le sommet d'une montagne.",
        "Le plongeur explore le fond de l'océan.",
        "L'astronome observe les étoiles avec un télescope.",
        "L'architecte dessine les plans d'un bâtiment.",
        "Le sculpteur façonne une statue en argile.",
        "Le danseur interprète un ballet gracieux.",
        "Le pâtissier décore un gâteau d'anniversaire.",
        "Le viticulteur récolte les raisins dans la vigne.",
        "Le surfeur glisse sur une vague géante.",
        "L'alpiniste escalade une paroi rocheuse.",
        "Le pêcheur lance sa ligne dans une rivière tranquille.",
        "Le chanteur interprète une chanson émouvante.",
        "Le professeur enseigne un cours passionnant.",
        "Le médecin ausculte un patient dans son cabinet.",
        "L'ingénieur conçoit un projet innovant.",
        "Le journaliste rédige un article d'actualité.",
        "Le pilote décolle avec son avion sur la piste.",
        "Le chercheur analyse des données dans son laboratoire.",
        "Le boulanger cuit du pain dans un four chaud.",
        "L'étudiant révise ses examens à la bibliothèque.",
        "Le couple danse lors d'un mariage romantique.",
        "L'enfant souffle les bougies de son gâteau d'anniversaire.",
        "La famille pique-nique dans un parc ensoleillé.",
        "Les amis discutent autour d'un café en terrasse.",
        "L'artiste expose ses œuvres dans une galerie.",
        "L'écrivain dédicace son livre dans une librairie.",
        "Les musiciens répètent un concert dans un studio.",
        "L'athlète court un marathon en ville.",
        "L'acteur répète une scène au théâtre.",
        "Le réalisateur tourne un film sur un plateau.",
        "Le chef cuisine un plat dans un restaurant.",
        "Le voyageur explore une ville inconnue.",
        "Le guide raconte une histoire dans un musée.",
        "Le pilote atterrit avec son avion à l'aéroport.",
        "Le chercheur découvre un remède dans son laboratoire.",
        "Le chat siamois miaule fort, réclamant de l'attention.",
        "Le chien berger allemand court à travers les champs fleuris.",
        "L'enfant dessine à la craie sur le trottoir devant la maison.",
        "L'abeille butine la fleur de lavande, récoltant son nectar.",

        # Nouveaux Ajouts - Nature et Météo
        "Un brouillard épais recouvre la vallée, masquant les maisons.",
        "La rosée du matin perle sur l'herbe, brillant au soleil levant.",
        "Un arc-en-ciel apparaît après la pluie, formant un pont coloré.",
        "Un orage éclate dans le ciel sombre, illuminant les nuages.",
        "La neige tombe doucement, recouvrant le paysage d'un manteau blanc.",
        "Un vent violent secoue les arbres, faisant tourbillonner les feuilles.",
        "La grêle frappe le toit de la maison, produisant un bruit assourdissant.",
        "Le soleil se couche derrière l'horizon, peignant le ciel de couleurs chaudes.",
        "Une brise légère rafraîchit l'air, apportant un soulagement bienvenu.",
        "Le feu crépite dans la cheminée du salon, diffusant une chaleur agréable.",
        "La cascade rugit du haut de la montagne rocheuse, créant un spectacle impressionnant.",
        "Un ruisseau serpente à travers la forêt calme, murmurant doucement.",
        "Le lac reflète les montagnes et le ciel, créant un miroir parfait.",
        "Un éclair zèbre le ciel pendant la nuit.",
        "Le tsunami a dévasté la côte après le séisme.",
        "L'éruption volcanique projette des cendres sur des kilomètres.",
        "L'aurore boréale danse dans le ciel polaire, offrant un spectacle magique.",

        # Nouveaux Ajouts - Activités et Objets
        "L'ordinateur portable affiche un écran bleu, signalant une erreur système.",
        "Le téléphone portable sonne dans le sac, indiquant un appel entrant.",
        "La clé USB contient des données importantes, sauvegardées avec soin.",
        "Le stylo bille écrit sur le papier blanc, laissant une trace d'encre.",
        "La fenêtre ouverte laisse entrer l'air frais, renouvelant l'atmosphère.",
        "La porte claque à cause du vent, produisant un bruit soudain.",
        "Les rideaux flottent devant la fenêtre ouverte, dansant avec la brise.",
        "Le miroir reflète la lumière de la pièce, agrandissant l'espace.",
        "L'horloge murale indique l'heure, rythmant le passage du temps.",
        "Le tapis persan décore le sol du salon, ajoutant une touche d'élégance.",
        "Le vase en cristal contient des fleurs fraîches, apportant de la couleur.",
        "La bougie parfumée brûle sur la table basse, diffusant un arôme agréable.",
        "La théière siffle dans la cuisine, annonçant que l'eau est chaude.",
        "La pizza chaude sort du four, dégageant une odeur appétissante.",
        "Le verre de vin rouge repose sur la table de la salle à manger, attendant d'être dégusté.",
        "La télévision diffuse le match de football en direct.",
        "Le réfrigérateur conserve les aliments au frais.",
        "La machine à laver nettoie les vêtements sales.",
        "L'aspirateur aspire la poussière du tapis.",
        "Le livre de recette explique comment préparer le plat.",

        # Nouveaux Ajouts - Émotions et Sensations
        "Une joie intense illumine son visage, exprimant un bonheur profond.",
        "Une tristesse profonde l'envahit, lui serrant le cœur.",
        "Une colère sourde gronde en lui, menaçant d'exploser.",
        "La peur panique le glace, l'empêchant de bouger.",
        "Une surprise agréable l'émerveille, lui faisant écarquiller les yeux.",
        "Un calme serein l'apaise, dissipant son stress.",
        "Une excitation vibrante le parcourt, le faisant frissonner.",
        "Une douleur aiguë lui lance dans le bras, le faisant grimacer.",
        "Une fatigue intense lui pèse sur les paupières, l'invitant au repos.",
        "Une odeur agréable lui chatouille les narines, stimulant son appétit.",
        "Un goût amer persiste dans sa bouche, laissant une sensation désagréable.",
        "Un son strident lui agresse les oreilles, le faisant sursauter.",
        "Un toucher doux le réconforte, apaisant sa peau.",
        "Un silence pesant envahit la pièce, créant une atmosphère tendue.",
        "La nostalgie l'envahit en se souvenant du passé.",
        "La curiosité pique l'intérêt de l'enfant pour le monde.",
        "La fierté gonfle sa poitrine après sa réussite.",
        "La honte fait rougir ses joues après son erreur.",
        "La frustration monte face à cet échec.",

        #Nouveaux ajouts - Actions plus complexes
        "L'étudiant surligne un passage important dans son livre avec un marqueur jaune.",
        "Le programmeur débogue le code informatique, cherchant la source de l'erreur.",
        "Le scientifique mélange des produits chimiques dans son laboratoire, observant les réactions.",
        "L'ouvrier construit un mur en briques, assemblant chaque pièce avec soin.",
        "Le conducteur de bus conduit les passagers à travers la ville, respectant les arrêts.",
        "L'infirmière administre une piqûre au patient, avec douceur et précision.",
        "Les pompiers éteignent l'incendie du bâtiment, luttant contre les flammes.",
        "Le policier dirige la circulation au carrefour, assurant la sécurité des usagers.",
        "Le serveur prend la commande des clients au restaurant, notant leurs choix.",
        "Le caissier scanne les articles au supermarché, enregistrant les achats.",
        "Le bibliothécaire range les livres sur les étagères, classant les ouvrages.",
        "Le comptable vérifie les factures au bureau, s'assurant de leur exactitude.",
        "Le traducteur traduit le document d'une langue étrangère, transmettant le sens.",
        "Le designer crée un logo pour l'entreprise, imaginant un visuel attractif.",
        "Le développeur web code le site internet, utilisant différents langages.",
        "L'agriculteur laboure son champ avec un tracteur.",
        "Le mécanicien répare le moteur de la voiture en panne.",
        "L'électricien installe le câblage dans la nouvelle maison.",
        "Le plombier répare la fuite d'eau dans la cuisine.",
        "Le juge rend son verdict au tribunal après délibération.",

        # Ajouts avec pronoms personnels (simples)
        "Je marche dans le parc.",
        "Tu lis un livre.",
        "Il joue au football.",
        "Elle chante une chanson.",
        "Nous mangeons une pizza.",
        "Vous regardez un film.",
        "Ils parlent à leurs amis.",
        "Elles dansent à la fête.",
        "On va au cinéma.",

        # Ajouts avec pronoms personnels (un peu plus complexes)
        "Je pense aux vacances.",
        "Tu espères qu'il fera beau.",
        "Il croit en sa réussite.",
        "Elle veut voyager.",
        "Nous devons travailler.",
        "Vous pouvez nous aider.",
        "Ils savent la réponse.",
        "Elles aiment la musique.",
        "On doit partir bientôt.",

        # Ajouts avec pronoms personnels (complexes)
        "Je crois que tu vas réussir.",
        "Tu sais où il est allé.",
        "Il pense à ce qu'elle a dit.",
        "Elle espère que nous allons venir.",
        "Nous voulons que vous compreniez.",
        "Vous savez comment ils ont fait ça.",
        "Ils ont dit qu'elles arriveraient bientôt.",
        "Elles pensent à ce qu'on leur a dit.",
        "On espère que tout va bien se passer.",
        "Je me souviens de toi avec émotion.",
        "Tu te demandes pourquoi il est parti.",
        "Il se prépare pour son examen de demain.",
        "Elle se maquille avant de sortir en soirée.",
        "Nous nous rencontrons au café du coin.",
        "Vous vous inquiétez pour rien, tout ira bien.",
        "Ils se disputent souvent, mais s'aiment beaucoup.",
        "Elles se promènent dans le parc au coucher du soleil.",
        "On se demande ce qui se passe, il y a beaucoup de bruit.",

        # Pronoms relatifs et compléments
        "Le livre que je lis est très intéressant.",
        "Le film que tu as regardé hier soir était émouvant.",
        "La personne à qui il parle est son professeur.",
        "La chanson qu'elle chante est magnifique.",
        "L'endroit où nous allons est secret.",
        "La raison pour laquelle vous partez est compréhensible.",
        "La chose dont ils parlent est confidentielle.",
        "Le moment où elles arriveront est incertain.",
        "La façon dont on fait cela est très simple.",
        "La fille avec qui je parlais est très sympathique.",
        "Le garçon dont tu te moques est en fait très gentil.",
        "Le problème auquel il pense est difficile à résoudre.",
        "La solution qu'elle a trouvée est ingénieuse.",
        "La ville d'où nous venons est très belle.",
        "Le pays où vous vivez semble fascinant.",
        "Les personnes avec qui ils travaillent sont compétentes.",
        "Les amis à qui elles écrivent sont loin.",
        "Le sujet sur lequel on discute est passionnant.",

        #Pronoms démonstratifs
        "Celui-ci est mon livre, ne le prends pas.",
        "Celle-là est ta voiture, elle est très belle.",
        "Ceux-ci sont mes amis, je te les présente.",
        "Celles-là sont leurs affaires, il faut les ranger.",
        "Ceci est important, écoute attentivement.",
        "Cela est intéressant, dis-m'en plus.",
        "Ceux que je préfère sont les plus grands.",
        "Celles que tu as choisies sont parfaites.",

        #Pronoms possessifs
        "Le mien est bleu.",
        "Le tien est plus grand.",
        "Le sien fonctionne bien.",
        "La sienne est rouge.",
        "Le nôtre arrivera demain.",
        "Le vôtre semble neuf.",
        "Les leurs sont sur la table.",
        "La mienne est en retard.",
        "Les tiens sont ici.",
        "Les siennes ont été perdues.",

        # ============================================
        # == AJOUTS PRÉEXISTANTS - Temps Verbaux ==
        # ============================================
        # Passé Composé
        "Le chat a mangé du poisson hier.",
        "Nous sommes arrivés en retard ce matin.",
        "Elle a lu ce livre la semaine dernière.",
        "Ils ont fini leur travail.",
        "J'ai déjà vu ce film.",
        "Tu es allé au magasin ?",

        # Imparfait
        "L'enfant jouait souvent dans le jardin.",
        "Il faisait beau quand nous sommes sortis.",
        "Les oiseaux chantaient tous les matins.",
        "Je pensais à toi hier soir.",
        "Vous habitiez ici avant ?",

        # Futur Simple
        "Nous visiterons le musée demain.",
        "Il pleuvra plus tard.",
        "Tu recevras une lettre bientôt.",
        "Elles partiront en vacances cet été.",
        "Je finirai ce projet la semaine prochaine.",

        # Conditionnel Présent
        "J'aimerais voyager plus souvent.",
        "Il pourrait t'aider si tu lui demandais.",
        "Nous préférerions rester à la maison.",
        "Serait-il possible de venir ?",
        "Vous devriez être plus prudent.",

        # Subjonctif Présent
        "Il faut que tu étudies plus.",
        "Je veux que vous soyez heureux.",
        "Bien qu'il soit fatigué, il travaille.",
        "Avant que nous partions, disons au revoir.",
        "Explique bien pour qu'ils comprennent.",

        # ============================================
        # == AJOUTS PRÉEXISTANTS - Négation ==
        # ============================================
        "Le chat ne dort pas sur le canapé.",
        "Je n'ai jamais vu cet oiseau.",
        "Il n'a plus faim.",
        "Nous ne comprenons rien.",
        "Vous ne parlez à personne ?",
        "Elle n'aime ni le chocolat ni le café.",

        # ============================================
        # == AJOUTS PRÉEXISTANTS - Interrogation ==
        # ============================================
        "Où le chien court-il ?",
        "Est-ce que l'enfant joue dans le jardin ?",
        "Pourquoi l'oiseau chante-t-il le matin ?",
        "Quand lirez-vous ce livre ?",
        "Comment fait-il ça ?",
        "Qui est arrivé en premier ?",
        "Tu aimes ce film ?",

        # ============================================
        # == AJOUTS PRÉEXISTANTS - Conjonctions ==
        # ============================================
        "Le chat dort et le chien veille.",
        "Il pleut, mais nous sortirons quand même.",
        "Tu es fatigué car tu as travaillé tard.",
        "Je reste ici parce que j'attends un ami.",
        "Appelle-moi quand tu arriveras.",
        "S'il fait beau, nous irons à la plage.",
        "Il lit pendant qu'elle cuisine.",
        "Bien que le livre soit long, il est intéressant.",
        "Tu peux choisir le fromage ou le dessert.",

        # ============================================
        # == AJOUTS PRÉEXISTANTS - Adverbes ==
        # ============================================
        "Le chien court rapidement dans le parc.",
        "L'oiseau chante magnifiquement.",
        "Nous arriverons bientôt.",
        "Il pleut beaucoup aujourd'hui.",
        "Parle plus doucement, s'il te plaît.",
        "Elle travaille toujours ici.",
        "L'enfant mange lentement sa soupe.",

        # ============================================
        # == AJOUTS PRÉEXISTANTS - Voix Passive ==
        # ============================================
        "Le poisson est mangé par le chat.",
        "La lettre a été écrite par Marie hier.",
        "La maison a été construite en 2020.",
        "Les règles doivent être respectées par tous.",

        # ============================================
        # == NOUVEAUX AJOUTS - Divers Complexité ==
        # ============================================
        "Malgré la pluie, nous sommes sortis.",
        "Grâce à ton aide, tu as réussi.",
        "L'homme marche lentement avec une canne.",
        "Voyager en avion est rapide.",
        "Aimer lire est une bonne chose.",
        "La voiture garée devant la maison est rouge.",
        "Les fleurs plantées par le jardinier sentent bon.",
        "C'est important de comprendre cette leçon.",
        "Il est facile de faire une erreur.",
        "Il faut donner à manger au chat affamé.",
        "Il m'a expliqué le problème clairement.",
        "Elle lui a offert un cadeau surprise.",
        "Penser à l'avenir le rend anxieux.",
        "Voir les enfants jouer me rend heureux.",

        # ============================================
        # == NOUVEAUX AJOUTS - Comparaison et Superlatif ==
        # ============================================
        "Le chat est plus rapide que la tortue.",
        "Le chien court aussi vite que le cheval.",
        "Il travaille moins que son collègue.",
        "Ce livre est le plus intéressant que j'ai lu.",
        "Elle parle le moins fort de tout le groupe.",
        "C'est la meilleure solution possible.",
        "Il a fait le pire choix de sa vie.",

        # ============================================
        # == NOUVEAUX AJOUTS - Discours Rapporté (simple) ==
        # ============================================
        "Il a dit qu'il était fatigué.",
        "Elle a demandé si nous viendrions.",
        "Ils ont demandé où nous allions.",
        "Je pense qu'il pleuvra demain.",

        # ============================================
        # == NOUVEAUX AJOUTS - Vocabulaire Spécifique (Tech, Science, Art) ==
        # ============================================
        "L'algorithme trie les données efficacement.",
        "Le télescope spatial observe une galaxie lointaine.",
        "La molécule d'ADN contient l'information génétique.",
        "Le peintre utilise une palette de couleurs vives.",
        "Le sculpteur taille le marbre pour créer une statue.",
        "Le compositeur écrit une symphonie pour l'orchestre.",
        "Le réseau neuronal apprend à partir des images.",
        "La photosynthèse convertit la lumière en énergie.",
        "Le danseur exécute une chorégraphie complexe avec grâce.",
    ]
}



df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# --- Chargement du modèle et du tokenizer ---
model_name = "dbddv01/gpt2-french-small"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# --- Ajout du token de padding (IMPORTANT) ---
# GPT-2 n'a pas de token de padding par défaut, il faut l'ajouter
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # On utilise le token de fin de séquence comme padding
    model.config.pad_token_id = model.config.eos_token_id


# --- Fonction de tokenisation (adaptée pour la génération de texte) ---
def preprocess_function(examples):
    inputs = examples['keywords']
    targets = examples['sentence']

    # Concaténer les keywords et la phrase cible, avec un séparateur
    #  (très important pour la génération !)
    concatenated_examples = [
        f"{input} {tokenizer.eos_token} {target}"
        for input, target in zip(inputs, targets)
    ]  # Ajout du token eos_token entre input et target

    # Tokeniser le tout
    model_inputs = tokenizer(
        concatenated_examples,
        truncation=True,
        padding='max_length',
        max_length=256,  # Augmente la longueur maximale si nécessaire
        return_tensors="pt" # Return pytorch tensor
    )

    # Les labels sont les mêmes que les inputs (pour la génération)
    model_inputs["labels"] = model_inputs["input_ids"].clone()

    return model_inputs

# --- Appliquer la tokenisation ---
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# --- Créer un DatasetDict ---
dataset_dict = DatasetDict({
    'train': tokenized_dataset
})

if 'validation' not in dataset_dict.keys():
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    dataset_dict['train'] = split_dataset['train']
    dataset_dict['validation'] = split_dataset['test']

train_dataset = dataset_dict['train']
eval_dataset = dataset_dict['validation']

# --- Data Collator (IMPORTANT pour GPT-2) ---
#  Il gère le padding et la création des tenseurs pour l'entraînement
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # mlm=False car on fait de la *génération*, pas du MLM
)

# --- Configuration de l'entraînement ---
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,  # Ajuste si nécessaire
    per_device_train_batch_size=4, # Réduire si tu as des erreurs de mémoire
    per_device_eval_batch_size=4,  # Réduire si tu as des erreurs de mémoire
    num_train_epochs=20, # Augmenter le nombre d'epochs pour un meilleur résultat
    weight_decay=0.01,
    push_to_hub=False, # Mettre True pour envoyer le model sur hugging face hub
)

# --- Initialisation du Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,  # Utilise le data collator
)

# --- Entraînement ---
trainer.train()

# --- Sauvegarde ---
model.save_pretrained("./gpt2_french_finetuned")
tokenizer.save_pretrained("./gpt2_french_finetuned")