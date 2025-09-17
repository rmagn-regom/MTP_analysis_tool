import os
import pandas as pd
import streamlit as st

# =====================
# Chargement CSV
# =====================
def charger_csv(file):
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
            'IDPASSAGE': str
        }
        return pd.read_csv(file, sep=';', dtype=dtype_dict, low_memory=False)
    except Exception as e:
        st.error(f"Erreur chargement CSV: {e}")
        return None

# =====================
# En-tête
# =====================
logo_path = os.path.join(os.path.dirname(__file__), 'logo.png')
if os.path.exists(logo_path):
    st.image(logo_path)
st.title("MTP Analysis Tool")

# =====================
# Session state
# =====================
for k,v in {
    'uploaded_file_id': None,
    'months_selected': [],
    'months_validated': False,
    'days_selected': [],
    'analysis_requested': False,
}.items():
    st.session_state.setdefault(k, v)

# =====================
# 1. Upload fichier
# =====================
uploaded_file = st.file_uploader("1. Fichier CSV", type=['csv'], key='file_uploader_main')
if uploaded_file is None:
    st.stop()

file_id = getattr(uploaded_file, 'name', None)
if st.session_state.uploaded_file_id != file_id:
    # reset workflow si nouveau fichier
    st.session_state.uploaded_file_id = file_id
    st.session_state.months_selected = []
    st.session_state.months_validated = False
    st.session_state.days_selected = []
    st.session_state.analysis_requested = False

df_raw = charger_csv(uploaded_file)
if df_raw is None or df_raw.empty:
    st.warning("Fichier vide ou illisible.")
    st.stop()

# Parse dates
if 'DTE' in df_raw.columns:
    df_raw['DTE'] = pd.to_datetime(df_raw['DTE'], errors='coerce', dayfirst=True)
    df_raw = df_raw.dropna(subset=['DTE'])
else:
    st.error("Colonne 'DTE' absente: impossible de poursuivre.")
    st.stop()

# =====================
# 2. Sélection des mois (obligatoire)
# =====================
mois_disponibles = sorted(df_raw['DTE'].dt.to_period('M').astype(str).unique())
with st.form('form_mois'):
    sel = st.multiselect("2. Choix des mois", mois_disponibles, default=st.session_state.months_selected)
    valider = st.form_submit_button("Valider les mois")
    if valider:
        st.session_state.months_selected = sel
        st.session_state.months_validated = True if sel else False
        st.session_state.days_selected = []  # reset jours si changement

if not st.session_state.months_validated:
    st.info("Sélectionnez et validez au moins un mois pour continuer.")
    st.stop()

# =====================
# 3. Sélection des jours (optionnel)
# =====================
df_months = df_raw[df_raw['DTE'].dt.to_period('M').astype(str).isin(st.session_state.months_selected)].copy()
jours_disponibles = sorted(df_months['DTE'].dt.date.unique())
jours_selectionnes = st.multiselect("3. (Option) Sélection de jours spécifiques", jours_disponibles, default=st.session_state.days_selected, key='jours_multiselect')
st.session_state.days_selected = jours_selectionnes

# =====================
# 4. Bouton lancer analyse
# =====================
if st.button("4. Lancer l'analyse"):
    st.session_state.analysis_requested = True

if not st.session_state.analysis_requested:
    st.stop()

# =====================
# Construction df_final selon filtres
# =====================
if st.session_state.days_selected:
    df_final = df_months[df_months['DTE'].dt.date.isin(st.session_state.days_selected)].copy()
else:
    df_final = df_months.copy()

if df_final.empty:
    st.warning("Aucune donnée après filtrage mois/jours.")
    st.stop()

# Ajout colonne Mois standard
df_final['Mois'] = df_final['DTE'].dt.to_period('M').astype(str)

# =====================
# Tabs (buffer optionnel si jours sélectionnés)
# =====================
tab_names = [
    "Vue d'ensemble",
    "Sorties automatiques",
    "Acquisition",
    "TRI",
    "Destinations"
]
include_buffer = len(st.session_state.days_selected) > 0
if include_buffer:
    tab_names.append('Buffer')

tabs = st.tabs(tab_names)
tab0, tab1, tab2, tab3, tab4 = tabs[:5]
tab_buffer = tabs[5] if include_buffer else None

with tab0:
    st.subheader("Vue d'ensemble")
    st.write(f"Lignes filtrées: {len(df_final):,}")
    st.write("Mois sélectionnés:", ", ".join(st.session_state.months_selected))
    if st.session_state.days_selected:
        st.write("Jours sélectionnés:", ", ".join([d.strftime('%d/%m/%Y') for d in st.session_state.days_selected]))
    st.dataframe(df_final.head(50))

with tab1:
    st.subheader("Sorties automatiques")
    # Placeholder simple: répartition par Mois
    if 'LIBELLEFAMILLEDESTMACHINE' in df_final.columns:
        counts = df_final.groupby(['Mois','LIBELLEFAMILLEDESTMACHINE']).size().unstack(fill_value=0)
        st.write("Répartition (nb lignes) par machine et mois")
        st.dataframe(counts)
    else:
        st.info("Colonne 'LIBELLEFAMILLEDESTMACHINE' absente.")

with tab2:
    st.subheader("Analyse Acquisition")
    req = {'IDPNEUREFERENCE','DTE','HEUREACQUISITION'}
    if req.issubset(df_final.columns):
        # Inclure 'Mois' pour les agrégations ultérieures
        work = df_final[['IDPNEUREFERENCE','DTE','HEUREACQUISITION','Mois']].copy()
        def to_td(x):
            if pd.isna(x): return pd.NaT
            s=str(x).strip()
            if len(s)==8 and s.count(':')==2:
                try: return pd.to_timedelta(s)
                except: return pd.NaT
            return pd.NaT
        work['H_ACQ'] = work['HEUREACQUISITION'].apply(to_td)
        work = work.dropna(subset=['H_ACQ'])
        work = work.sort_values(['DTE','H_ACQ','IDPNEUREFERENCE']).reset_index(drop=True)
        work['H_ACQ_PREV'] = work.groupby(['DTE'])['H_ACQ'].shift(1)
        work['DTE_PREV'] = work['DTE'].shift(1)
        work['TPS_CYCLE_AQUISITION'] = (work['H_ACQ'] - work['H_ACQ_PREV']).dt.total_seconds().round(2)
        work.loc[work['DTE']!=work['DTE_PREV'],'TPS_CYCLE_AQUISITION'] = None
        hhmm = work['H_ACQ'].dt.components['hours']*60 + work['H_ACQ'].dt.components['minutes']
        mask_sup = (work['TPS_CYCLE_AQUISITION']>2700) & (hhmm>=690) & (hhmm<=840)
        work = work[~mask_sup]
        if work.empty:
            st.warning("Aucune donnée acquisition exploitable après filtrage.")
        else:
            bins = [0,4,6,8,12,60,float('inf')]
            labels = ['0-4s','4-6s','6-8s','8-12s','12-60s','>60s']
            work['TPS_CYCLE_CAT'] = pd.cut(work['TPS_CYCLE_AQUISITION'], bins=bins, labels=labels, include_lowest=True)
            ctab = pd.crosstab(work['Mois'], work['TPS_CYCLE_CAT'], normalize='index')*100
            ctab = ctab.round(1)
            synth = work.groupby('Mois')['TPS_CYCLE_AQUISITION'].mean().round(2)
            ctab['Moyenne (s)'] = synth
            st.write("Distribution des temps de cycle acquisition (%)")
            st.dataframe(ctab)
    else:
        st.info("Colonnes nécessaires absentes pour l'analyse Acquisition.")

with tab3:
    st.subheader("Analyse TRI")
    required_cols = {'DTE','HEURETRIAGE','HEUREPRISEPNEU','TRIPOSTE'}
    if required_cols.issubset(df_final.columns):
        def h_to_td(h):
            try:
                if pd.isna(h): return pd.NaT
                s=str(h).strip()
                if len(s)==8 and s.count(':')==2: return pd.to_timedelta(s)
                if len(s)==7 and s.count(':')==2: return pd.to_timedelta('0'+s)
                return pd.NaT
            except: return pd.NaT
        work = df_final.copy()
        work['HEURETRIAGE'] = work['HEURETRIAGE'].apply(h_to_td)
        work['HEUREPRISEPNEU'] = work['HEUREPRISEPNEU'].apply(h_to_td)
        work['TPS_TRIAGE'] = (work['HEURETRIAGE'] - work['HEUREPRISEPNEU']).dt.total_seconds().round(2)
        work = work.sort_values(['TRIPOSTE','DTE','HEUREPRISEPNEU','IDPNEUREFERENCE']).reset_index(drop=True)
        work['HEURETRIAGE_PREV'] = work.groupby(['TRIPOSTE','DTE'])['HEURETRIAGE'].shift(1)
        work['DTE_PREV'] = work['DTE'].shift(1)
        work['TRIPOSTE_PREV'] = work['TRIPOSTE'].shift(1)
        work['TPS_MANUT'] = (work['HEUREPRISEPNEU'] - work['HEURETRIAGE_PREV']).dt.total_seconds().round(2)
        mask_cycle = (work['DTE']!=work['DTE_PREV']) | (work['TRIPOSTE']!=work['TRIPOSTE_PREV'])
        work.loc[mask_cycle,'TPS_MANUT'] = None
        work['TPS_CYCLE_TRI'] = work['TPS_MANUT'] + work['TPS_TRIAGE']
        synth = work.groupby(['Mois','TRIPOSTE'])[['TPS_TRIAGE','TPS_CYCLE_TRI','TPS_MANUT']].agg(['count','mean']).reset_index()
        for col in ['TPS_TRIAGE','TPS_CYCLE_TRI','TPS_MANUT']:
            synth[(col,'mean')] = synth[(col,'mean')].round(2)
        synth.columns = ['Mois','TRIPOSTE','Nb_TRIAGE','Moy_TRIAGE','Nb_CYCLE_TRI','Moy_CYCLE_TRI','Nb_MANUT','Moy_MANUT']
        synth_pivot = synth.pivot(index='Mois', columns='TRIPOSTE', values=['Moy_MANUT','Moy_TRIAGE','Moy_CYCLE_TRI'])
        synth_count = synth.pivot(index='Mois', columns='TRIPOSTE', values=['Nb_TRIAGE'])
        fusion_table = pd.concat({'Moy': synth_pivot, 'Nb': synth_count}, axis=1)
        st.write('Synthèse mensuelle par TRIPOSTE')
        st.dataframe(fusion_table)
        if 'LIBELLEFAMILLEDESTCHOISISSEUR' in work.columns:
            fam = work.groupby('LIBELLEFAMILLEDESTCHOISISSEUR')['TPS_TRIAGE'].mean().round(2).reset_index()
            fam = fam.rename(columns={'TPS_TRIAGE':'Moyenne TPS_TRIAGE (s)'})
            st.write("Moyenne TPS_TRIAGE par famille")
            st.dataframe(fam)
    else:
        st.info("Colonnes TRI manquantes.")

with tab4:
    st.subheader("Destinations")
    if {'LIBELLEFAMILLEDESTMACHINE','LIBELLEFAMILLEDESTCHOISISSEUR'}.issubset(df_final.columns):
        work = df_final.copy()
        crosstab_pct = pd.crosstab(
            [work['Mois'], work['LIBELLEFAMILLEDESTMACHINE']],
            work['LIBELLEFAMILLEDESTCHOISISSEUR'],
            normalize='index'
        ) * 100
        crosstab_pct = crosstab_pct.round(1)
        crosstab_count = pd.crosstab(
            [work['Mois'], work['LIBELLEFAMILLEDESTMACHINE']],
            work['LIBELLEFAMILLEDESTCHOISISSEUR']
        )
        merged = pd.DataFrame(index=crosstab_pct.index)
        for col in crosstab_pct.columns:
            merged[("LIBELLEFAMILLEDESTCHOISISSEUR", f"{col} (%)")] = crosstab_pct[col]
            merged[("LIBELLEFAMILLEDESTCHOISISSEUR", f"{col} (U)")] = crosstab_count[col]
        merged = merged.reset_index().rename(columns={'Mois':('Prop MTP','Mois'),'LIBELLEFAMILLEDESTMACHINE':('Prop MTP','Libellé')})
        new_cols = []
        for col in merged.columns:
            if isinstance(col, tuple):
                if col[0]=='LIBELLEFAMILLEDESTCHOISISSEUR':
                    new_cols.append(('Choix Trieur', col[1]))
                else:
                    new_cols.append(col)
            else:
                new_cols.append(('Prop MTP', col))
        merged.columns = pd.MultiIndex.from_tuples(new_cols)
        st.dataframe(merged, use_container_width=True)
    else:
        st.info("Colonnes nécessaires absentes pour l'analyse Destinations.")

if include_buffer and tab_buffer is not None:
    with tab_buffer:
        st.subheader("Simulation remplissage convoyeur")
        required_cols = {'DTE','HEUREACQUISITION','HEUREPRISEPNEU'}
        if not required_cols.issubset(df_final.columns):
            st.info("Colonnes nécessaires manquantes pour la simulation (besoin: DTE, HEUREACQUISITION, HEUREPRISEPNEU).")
        else:
            import matplotlib.pyplot as plt
            # Préparation heures en Timedelta
            def to_td(h):
                if pd.isna(h): return pd.NaT
                s=str(h).strip()
                if len(s)==8 and s.count(':')==2:
                    try: return pd.to_timedelta(s)
                    except: return pd.NaT
                if len(s)==7 and s.count(':')==2:
                    try: return pd.to_timedelta('0'+s)
                    except: return pd.NaT
                return pd.NaT
            df_buf = df_final.copy()
            df_buf['H_ACQ'] = df_buf['HEUREACQUISITION'].apply(to_td)
            df_buf['H_TRI'] = df_buf.get('HEUREPRISEPNEU', pd.Series([pd.NaT]*len(df_buf))).apply(to_td) if 'HEUREPRISEPNEU' in df_buf.columns else pd.NaT
            # Filtrer sur jours sélectionnés
            jours_sim = sorted(set(st.session_state.days_selected))
            if not jours_sim:
                st.info("Sélectionnez au moins un jour pour voir la simulation.")
            for jour in jours_sim:
                df_day = df_buf[df_buf['DTE'].dt.date == jour].copy()
                st.markdown(f"**Jour : {jour.strftime('%d/%m/%Y')}**")
                if df_day.empty:
                    st.info("Aucune donnée pour ce jour.")
                    continue
                df_day = df_day.sort_values('H_ACQ').dropna(subset=['H_ACQ']).reset_index(drop=True)
                events = []  # (time, +1/-1)
                positions = [None]*24  # Fin d'occupation prévue par position
                for _, row in df_day.iterrows():
                    t_in = row['H_ACQ']
                    if pd.isna(t_in):
                        continue
                    t_out_real = row['H_TRI'] if pd.notna(row['H_TRI']) else None
                    # Cherche première position libre
                    pos = 0
                    while pos < 24 and positions[pos] is not None and positions[pos] <= t_in:
                        # Libérer positions dont temps est écoulé
                        if positions[pos] <= t_in:
                            positions[pos] = None
                        pos += 1
                    # Re-scan pour première case vide
                    pos = 0
                    while pos < 24 and positions[pos] is not None:
                        pos += 1
                    if pos < 24:
                        # Placement direct
                        if t_out_real and t_out_real > t_in:
                            t_out = t_out_real
                        else:
                            # Durée simulée: 1.5s * pos (ou 6 si valorisé Non)
                            if row.get('LIBELLEVALORISEDESTMACHINE','') == 'Non':
                                pos_sim = 6
                                t_out = t_in + pd.to_timedelta(1.5*pos_sim, unit='s')
                                pos = pos_sim
                            else:
                                t_out = t_in + pd.to_timedelta(1.5*pos, unit='s')
                        events.append((t_in, +1))
                        events.append((t_out, -1))
                        positions[pos] = t_out
                    else:
                        # Convoyeur plein: attendre libération la plus proche
                        active_times = [t for t in positions if t is not None]
                        if not active_times:
                            continue
                        t_next = min(active_times)
                        pos_libre = positions.index(t_next)
                        t_in2 = max(t_in, t_next) + pd.to_timedelta(1.75, unit='s')
                        if t_out_real and t_out_real > t_in2:
                            t_out = t_out_real
                        else:
                            if row.get('LIBELLEVALORISEDESTMACHINE','') == 'Non':
                                t_out = t_in2 + pd.to_timedelta(1.75*6, unit='s')
                                pos_libre = 6
                            else:
                                t_out = t_in2 + pd.to_timedelta(1.75*(23-pos_libre), unit='s')
                        events.append((t_in2, +1))
                        events.append((t_out, -1))
                        positions[pos_libre] = t_out
                if not events:
                    st.info("Aucun événement à simuler.")
                    continue
                # Construire timeline cumulée
                events.sort()
                timeline = []
                count = 0
                for t, delta in events:
                    count += delta
                    timeline.append((t, count))
                times = [t.total_seconds()/3600 for t,_ in timeline]
                values = [v for _,v in timeline]
                fig, ax = plt.subplots(figsize=(8,4))
                ax.step(times, values, where='post')
                ax.set_xlabel('Heure (décimal)')
                ax.set_ylabel('Nb pneus sur convoyeur')
                ax.set_title(f"Remplissage convoyeur {jour.strftime('%d/%m/%Y')}")
                ax.set_ylim(0,25)
                st.pyplot(fig)
