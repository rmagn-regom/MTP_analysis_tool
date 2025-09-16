import pandas as pd
import streamlit as st

# Charger le fichier CSV
def charger_csv(path):
    try:
        dtype_dict = {
            'IDPNEUREFERENCE': 'Int64',
            'TRIPOSTE': str,
            'INDICECHARGEJ': 'Int64',
            'LIBELLEMARQUE': str,
            'LIBELLEPROFIL': str,
            'LARGEUR': str,
            'SERIE': str,
            'JANTE': str,
            'INDICECHARGE': str,
            'INDICEVITESSE': str,
            'DOT': str,
            'IDBENNE': str,
            'BENNEREPERE': str,
            'IDPASSAGE': str,
        }
        df = pd.read_csv(path, sep=';', dtype=dtype_dict, low_memory=False)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du CSV : {e}")
        return None

# Calculs complémentaires (exemple : somme d'une colonne numérique)
def calculs_complementaires(df):
    pass  # Cette fonction n'est plus utilisée

# Interface Streamlit


# Affichage du logo après le titre
import os
logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
if os.path.exists(logo_path):
    st.image(logo_path)

st.title("MTP Analysis Tool")

fichier = st.file_uploader("Choisissez un fichier CSV", type=["csv"])
if fichier is not None:
    # Réinitialiser la sélection des mois à chaque nouveau fichier
    for k in ['mois_selectionnes', 'mois_selection_temp', 'mois_valide']:
        if k in st.session_state:
            del st.session_state[k]
    df = charger_csv(fichier)
    if df is not None:
        # Conversion de la date pour extraire les mois AVANT tout calcul
        df['DTE'] = pd.to_datetime(df['DTE'], format='%d/%m/%Y', errors='coerce')
        mois_disponibles = df['DTE'].dt.to_period('M').dropna().astype(str).unique()
        mois_disponibles = sorted(mois_disponibles)
        # Initialiser la sélection et la validation dans session_state
        if 'mois_selectionnes' not in st.session_state:
            st.session_state['mois_selectionnes'] = []
        if 'mois_valide' not in st.session_state:
            st.session_state['mois_valide'] = False
        if 'mois_selection_temp' not in st.session_state:
            st.session_state['mois_selection_temp'] = []

        # Multiselect temporaire
        mois_selection_temp = st.multiselect(
            "Sélectionnez les mois à analyser :",
            options=mois_disponibles,
            default=st.session_state['mois_selection_temp'],
            key='mois_multiselect_temp'
        )
        # Bouton de validation
        if st.button("Valider la sélection"):
            if mois_selection_temp:
                st.session_state['mois_selectionnes'] = mois_selection_temp
                st.session_state['mois_selection_temp'] = mois_selection_temp
                st.session_state['mois_valide'] = True
            else:
                st.warning("Veuillez sélectionner au moins un mois pour lancer l'analyse.")
        else:
            # Mettre à jour la variable temporaire à chaque interaction
            st.session_state['mois_selection_temp'] = mois_selection_temp

        # Affichage des tabs et calculs seulement si validation OK
        if st.session_state['mois_valide'] and st.session_state['mois_selectionnes']:
            # Filtrer le DataFrame sur les mois sélectionnés
            df = df[df['DTE'].dt.to_period('M').astype(str).isin(st.session_state['mois_selectionnes'])].copy()
            tab0, tab1, tab2, tab3, tab4 = st.tabs(["Aperçu des données", "Sorties automatiques", "Aquisition MTP", "TRI AB", "Analyse destinations"])

            with tab0:
                st.write("Aperçu des 50 premières lignes du fichier :")
                st.dataframe(df.head(50))

            with tab1:
                if 'LIBELLEVALORISEDESTMACHINE' in df.columns:
                    st.write("\n**Répartition des Sorties Automatiques :**")
                    counts = df['LIBELLEVALORISEDESTMACHINE'].value_counts()
                    def autopct_format(pct):
                        total = counts.sum()
                        val = int(round(pct*total/100.0))
                        return f'{pct:.1f}%\n({val:,})'.replace(',', ' ').replace('\xa0', ' ')
                    st.pyplot(counts.plot.pie(autopct=autopct_format, ylabel='').get_figure())
                else:
                    st.warning("La colonne 'LIBELLEVALORISEDESTMACHINE' n'existe pas dans le fichier CSV.")

            with tab2:
                required_cols = {'IDPNEUREFERENCE', 'DTE', 'HEUREACQUISITION'}
                if required_cols.issubset(df.columns):
                    with st.spinner('Calcul en cours...'):
                        def heure_to_timedelta(h):
                            try:
                                if pd.isna(h):
                                    return pd.NaT
                                if isinstance(h, pd.Timedelta):
                                    return h
                                h_str = str(h).strip()
                                # Si format hh:mm:ss strict
                                if len(h_str) == 8 and h_str.count(":") == 2:
                                    return pd.to_timedelta(h_str)
                                # Si format h:mm:ss, compléter à gauche
                                if len(h_str) == 7 and h_str.count(":") == 2:
                                    return pd.to_timedelta('0'+h_str)
                                return pd.NaT
                            except:
                                return pd.NaT
                        df['HEUREACQUISITION'] = df['HEUREACQUISITION'].apply(heure_to_timedelta)
                        # Créer une colonne datetime complète pour l'acquisition
                        df['DH_ACQUISITION'] = df['DTE'] + df['HEUREACQUISITION']
                        # Créer la colonne Mois AVANT tout usage
                        df['Mois'] = df['DTE'].dt.to_period('M').astype(str)
                        # Tri
                        df = df.sort_values(['IDPNEUREFERENCE', 'DH_ACQUISITION']).reset_index(drop=True)
                        # Décalage des valeurs précédentes
                        df['DH_ACQUISITION_PREV'] = df['DH_ACQUISITION'].shift(1)
                        df['IDPNEUREFERENCE_PREV'] = df['IDPNEUREFERENCE'].shift(1)
                        df['DTE_PREV'] = df['DTE'].shift(1)
                        # Calcul du temps entre deux acquisitions (en secondes, 2 décimales)
                        df['TPS_CYCLE_AQUISITION'] = (df['DH_ACQUISITION'] - df['DH_ACQUISITION_PREV']).dt.total_seconds().round(2)
                        # Si IDPNEUREFERENCE ou DTE change, mettre NaN
                        mask = (df['DTE'] != df['DTE_PREV'])
                        df.loc[mask, 'TPS_CYCLE_AQUISITION'] = None
                        # Exclure les temps de cycle > 45 min (2700s) ET dont l'heure d'acquisition est entre 11h30 et 14h00
                        mask_exclure = (
                            (df['TPS_CYCLE_AQUISITION'] > 2700)
                            & (df['HEUREACQUISITION'].notna())
                            & (df['HEUREACQUISITION'].dt.components['hours'] * 60 + df['HEUREACQUISITION'].dt.components['minutes'] >= 690)
                            & (df['HEUREACQUISITION'].dt.components['hours'] * 60 + df['HEUREACQUISITION'].dt.components['minutes'] <= 840)
                        )
                        df = df[~mask_exclure].copy()
                        # S'assurer que la colonne 'Mois' existe après filtrage
                        if 'Mois' not in df.columns:
                            df['Mois'] = df['DTE'].dt.to_period('M').astype(str)
                        # Vérifier si le DataFrame est vide après filtrage
                        if df.empty:
                            st.warning("Aucune donnée à afficher après filtrage (cycle > 45min entre 11h30 et 14h00 exclu).")
                        else:
                            bins = [0, 5, 10, 20, 60, float('inf')]
                            labels = ['0-5s', '6-10s', '11-20s', '21-60s', '>60s']
                            df['TPS_CYCLE_CAT'] = pd.cut(df['TPS_CYCLE_AQUISITION'], bins=bins, labels=labels, right=True, include_lowest=True)
                            crosstab = pd.crosstab(df['Mois'], df['TPS_CYCLE_CAT'], normalize='index') * 100
                            crosstab = crosstab.round(1)
                            synth = df.groupby('Mois')['TPS_CYCLE_AQUISITION'].mean().round(2)
                            crosstab['Moyenne (s)'] = synth
                            st.write('**Répartition des temps de cycle par intervalles et par mois (%)**')
                            st.dataframe(crosstab)
                else:
                    st.warning("Colonnes nécessaires manquantes : IDPNEUREFERENCE, DTE, HEUREACQUISITION")

            with tab3:                        
                # Vérifier la présence des colonnes nécessaires
                required_cols = {'DTE', 'HEURETRIAGE', 'HEUREPRISEPNEU', 'TRIPOSTE'}
                if required_cols.issubset(df.columns):
                        with st.spinner('Calcul en cours...'):
                            # Conversion des dates et heures
                            df['DTE'] = pd.to_datetime(df['DTE'], format='%d/%m/%Y', errors='coerce')
                            df['Mois'] = df['DTE'].dt.to_period('M').astype(str)
                            def heure_to_timedelta_str(h):
                                try:
                                    if pd.isna(h):
                                        return pd.NaT
                                    h_str = str(h).strip()
                                    # Si format hh:mm:ss strict
                                    if len(h_str) == 8 and h_str.count(":") == 2:
                                        return pd.to_timedelta(h_str)
                                    # Si format h:mm:ss, compléter à gauche
                                    if len(h_str) == 7 and h_str.count(":") == 2:
                                        return pd.to_timedelta('0'+h_str)
                                    return pd.NaT
                                except:
                                    return pd.NaT
                            df['HEURETRIAGE'] = df['HEURETRIAGE'].apply(heure_to_timedelta_str)
                            df['HEUREPRISEPNEU'] = df['HEUREPRISEPNEU'].apply(heure_to_timedelta_str)
                            # Calcul de la différence en secondes (float, 2 décimales)
                            df['TPS_TRIAGE'] = (df['HEURETRIAGE'] - df['HEUREPRISEPNEU']).dt.total_seconds().round(2)

                            # Calcul du TPS_CYCLE_TRI (même logique que TPS_CYCLE_AQUISITION, mais avec HEUREPRISEPNEU)
                            df = df.sort_values(['TRIPOSTE', 'DTE', 'HEUREPRISEPNEU', 'IDPNEUREFERENCE']).reset_index(drop=True)
                            # Décalage des valeurs précédentes pour le même TRIPOSTE et DTE
                            df['HEURETRIAGE_PREV'] = df.groupby(['TRIPOSTE', 'DTE'])['HEURETRIAGE'].shift(1)
                            df['TRIPOSTE_PREV'] = df['TRIPOSTE'].shift(1)
                            df['DTE_PREV'] = df['DTE'].shift(1)
                            # Calcul du temps entre deux prises pneu (en secondes)
                            df['TPS_MANUT'] = (df['HEUREPRISEPNEU'] - df['HEURETRIAGE_PREV']).dt.total_seconds().round(2)
                            # Si DTE ou TRIPOSTE change, mettre NaN
                            mask_cycle = (df['DTE'] != df['DTE_PREV']) | (df['TRIPOSTE'] != df['TRIPOSTE_PREV'])
                            df.loc[mask_cycle, 'TPS_MANUT'] = None

                            # Calcul du TPS_MANUT = TPS_CYCLE_TRI - TPS_TRIAGE
                            df['TPS_CYCLE_TRI'] = df['TPS_MANUT'] + df['TPS_TRIAGE']

                            # # Affichage du DataFrame avec les nouvelles colonnes
                            # st.write("Aperçu des calculs :")
                            # df_aff = df[['DTE','TRIPOSTE','HEUREPRISEPNEU','HEURETRIAGE','TPS_MANUT','TPS_TRIAGE','TPS_CYCLE_TRI']].copy()
                            # df_aff['HEUREPRISEPNEU'] = df_aff['HEUREPRISEPNEU'].apply(lambda x: str(x) if pd.notna(x) else '')
                            # df_aff['HEURETRIAGE'] = df_aff['HEURETRIAGE'].apply(lambda x: str(x) if pd.notna(x) else '')
                            # # Nettoyer le format pour n'afficher que hh:mm:ss
                            # df_aff['HEUREPRISEPNEU'] = df_aff['HEUREPRISEPNEU'].str.extract(r'(\d{1,2}:\d{2}:\d{2})')[0].fillna('')
                            # df_aff['HEURETRIAGE'] = df_aff['HEURETRIAGE'].str.extract(r'(\d{1,2}:\d{2}:\d{2})')[0].fillna('')
                            # st.dataframe(df_aff.head(50))

                            # Synthèse mensuelle par TRIPOSTE (TRIPOSTE en colonne, Moyenne (s) [Nb])
                            synth = df.groupby(['Mois','TRIPOSTE'])[['TPS_TRIAGE','TPS_CYCLE_TRI','TPS_MANUT']].agg(['count','mean']).reset_index()
                            for col in ['TPS_TRIAGE','TPS_CYCLE_TRI','TPS_MANUT']:
                                synth[(col,'mean')] = synth[(col,'mean')].round(2)
                            synth.columns = ['Mois','TRIPOSTE',
                                            'Nb_TRIAGE','Moy_TRIAGE',
                                            'Nb_CYCLE_TRI','Moy_CYCLE_TRI',
                                            'Nb_MANUT','Moy_MANUT']
                            synth_pivot = synth.pivot(index='Mois', columns='TRIPOSTE', values=['Moy_MANUT','Moy_TRIAGE','Moy_CYCLE_TRI'])
                            synth_count = synth.pivot(index='Mois', columns='TRIPOSTE', values=['Nb_TRIAGE'])
                            fusion_table = pd.concat({'Moy': synth_pivot, 'Nb': synth_count}, axis=1)
                            st.write('**Synthèse mensuelle par TRIPOSTE : Moyenne (s) et nombre de cas**')
                            st.dataframe(fusion_table)

                            # Moyenne de TPS_TRIAGE par LIBELLEFAMILLEDESTCHOISISSEUR
                            if 'LIBELLEFAMILLEDESTCHOISISSEUR' in df.columns:
                                st.write('**Moyenne de TPS_TRIAGE par LIBELLEFAMILLEDESTCHOISISSEUR**')
                                famille_triage = df.groupby('LIBELLEFAMILLEDESTCHOISISSEUR')['TPS_TRIAGE'].mean().round(2).reset_index()
                                famille_triage = famille_triage.rename(columns={'TPS_TRIAGE': 'Moyenne TPS_TRIAGE (s)'})
                                st.dataframe(famille_triage)
                            else:
                                st.info("Colonne 'LIBELLEFAMILLEDESTCHOISISSEUR' absente du fichier.")
                else:
                    st.warning("Colonnes nécessaires manquantes : DTE, HEURETRIAGE, HEUREPRISEPNEU, TRIPOSTE")

            with tab4:
                # Répartition mensuelle de LIBELLEFAMILLEDESTCHOISISSEUR par LIBELLEFAMILLEDESTMACHINE
                        if 'LIBELLEFAMILLEDESTMACHINE' in df.columns and 'LIBELLEFAMILLEDESTCHOISISSEUR' in df.columns:
                            st.write("**Répartition mensuelle (%) et quantité de Prop MTP pour chaque Choix Trieur**")
                            # Table de contingence pour les pourcentages
                            crosstab_pct = pd.crosstab(
                                [df['Mois'], df['LIBELLEFAMILLEDESTMACHINE']],
                                df['LIBELLEFAMILLEDESTCHOISISSEUR'],
                                normalize='index'
                            ) * 100
                            crosstab_pct = crosstab_pct.round(1)
                            # Table de contingence pour les effectifs
                            crosstab_count = pd.crosstab(
                                [df['Mois'], df['LIBELLEFAMILLEDESTMACHINE']],
                                df['LIBELLEFAMILLEDESTCHOISISSEUR']
                            )
                            # Fusionner les deux tables en concaténant les colonnes (ex: "VAL1 (%)", "VAL1 (N)")
                            # Construction d'un MultiIndex de colonnes pour affichage structuré
                            arrays = [[], []]
                            for col in crosstab_pct.columns:
                                arrays[0].extend(["LIBELLEFAMILLEDESTCHOISISSEUR"]*2)
                                arrays[1].extend([str(col) + ' (%)', str(col) + ' (U)'])
                            multi_columns = pd.MultiIndex.from_arrays(arrays)
                            merged = pd.DataFrame(index=crosstab_pct.index)
                            for col in crosstab_pct.columns:
                                merged[("LIBELLEFAMILLEDESTCHOISISSEUR", str(col) + ' (%)')] = crosstab_pct[col]
                                merged[("LIBELLEFAMILLEDESTCHOISISSEUR", str(col) + ' (U)')] = crosstab_count[col]
                            merged = merged.reset_index()
                            # Renommer les colonnes d'index pour affichage
                            merged = merged.rename(columns={
                                'Mois': ('Prop MTP', 'Mois'),
                                'LIBELLEFAMILLEDESTMACHINE': ('Prop MTP', 'Libellé'),
                            })
                            # Construction du MultiIndex final
                            new_cols = []
                            for col in merged.columns:
                                if isinstance(col, tuple):
                                    # Pour les colonnes de pourcentage/compte, remplacer le niveau 0 par 'Choix Trieur'
                                    if col[0] == 'LIBELLEFAMILLEDESTCHOISISSEUR':
                                        new_cols.append(('Choix Trieur', col[1]))
                                    elif col[0] == 'Prop MTP':
                                        new_cols.append(col)
                                    else:
                                        new_cols.append(col)
                                else:
                                    # Pour toute colonne d'index restante
                                    new_cols.append(('Prop MTP', col))
                            merged.columns = pd.MultiIndex.from_tuples(new_cols)
                            st.dataframe(merged, use_container_width=True)