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
    st.image(logo_path ,width=200)
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
    'conv_positions': 24,
    'conv_special': 6,
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
# 3. Sélection des jours + paramètres convoyeur (optionnels)
# =====================
df_months = df_raw[df_raw['DTE'].dt.to_period('M').astype(str).isin(st.session_state.months_selected)].copy()
jours_disponibles = sorted(df_months['DTE'].dt.date.unique())

col_jours, col_conf = st.columns([2,1])
with col_jours:
    jours_selectionnes = st.multiselect(
        "3. (Option) Sélection de jours spécifiques",
        jours_disponibles,
        default=st.session_state.days_selected,
        key='jours_multiselect'
    )
with col_conf:
    st.markdown("**Paramètres convoyeur**")
    conv_positions = st.number_input("Positions", min_value=1, max_value=200, value=st.session_state.conv_positions, step=1, key='conv_positions_input')
    conv_special = st.number_input("Position spéciale 'Non'", min_value=0, max_value=200, value=st.session_state.conv_special, step=1, key='conv_special_input')
    # Clamp si incohérence
    if conv_special >= conv_positions:
        st.warning("La position spéciale doit être < Positions. Ajustement automatique.")
        conv_special = max(0, conv_positions-1)
    st.session_state.conv_positions = conv_positions
    st.session_state.conv_special = conv_special

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
    st.subheader("Synthèse LIBELLEVALORISEDESTMACHINE")
    champ = 'LIBELLEVALORISEDESTMACHINE'
    if champ in df_final.columns:
        # Table effectifs
        tab_n = pd.crosstab(df_final['Mois'], df_final[champ], margins=True, margins_name='Total')
        # Table pourcentages
        tab_pct = pd.crosstab(df_final['Mois'], df_final[champ], normalize='index', margins=True, margins_name='Total')*100
        tab_pct = tab_pct.round(1)
        # Fusion pour affichage combiné, cellule par cellule
        def format_cell(n, pct):
            try:
                return f"{int(n)} ({pct:.1f}%)"
            except:
                return f"{n} ({pct}%)"
        tab_mix = tab_n.copy()
        # Pour éviter KeyError sur 'Total', n'itérer que sur l'intersection des index/colonnes
        idxs = set(tab_n.index) & set(tab_pct.index)
        cols = set(tab_n.columns) & set(tab_pct.columns)
        for idx in idxs:
            for col in cols:
                n = tab_n.at[idx, col]
                pct = tab_pct.at[idx, col]
                tab_mix.at[idx, col] = format_cell(n, pct)
        # Pour les totaux (lignes/colonnes hors intersection), afficher juste le nombre
        for idx in tab_n.index:
            for col in tab_n.columns:
                if (idx not in idxs) or (col not in cols):
                    tab_mix.at[idx, col] = str(tab_n.at[idx, col])
        st.write("Tableau croisé Mois x ", champ, " (N (pct %)) avec totaux")
        st.dataframe(tab_mix)
    else:
        st.info(f"Colonne '{champ}' absente.")

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
                total_positions = int(st.session_state.conv_positions)
                special_pos = int(st.session_state.conv_special)
                positions = [None]*total_positions  # Fin d'occupation prévue par position
                for _, row in df_day.iterrows():
                    t_in = row['H_ACQ']
                    if pd.isna(t_in):
                        continue
                    t_out_real = row['H_TRI'] if pd.notna(row['H_TRI']) else None
                    # Cherche première position libre
                    pos = 0
                    while pos < total_positions and positions[pos] is not None and positions[pos] <= t_in:
                        # Libérer positions dont temps est écoulé
                        if positions[pos] <= t_in:
                            positions[pos] = None
                        pos += 1
                    # Re-scan pour première case vide
                    pos = 0
                    while pos < total_positions and positions[pos] is not None:
                        pos += 1
                    if pos < total_positions:
                        # Placement direct
                        if t_out_real and t_out_real > t_in:
                            t_out = t_out_real
                        else:
                            # Durée simulée: 1.5s * pos (ou special_pos si valorisé Non)
                            if row.get('LIBELLEVALORISEDESTMACHINE','') == 'Non':
                                pos_sim = special_pos
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
                                t_out = t_in2 + pd.to_timedelta(1.75*special_pos, unit='s')
                                pos_libre = special_pos
                            else:
                                # distance dynamique jusqu'à dernière position
                                t_out = t_in2 + pd.to_timedelta(1.75*((total_positions-1)-pos_libre), unit='s')
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
                # Construction d'une série minute par minute pour un affichage en barres interactif
                import numpy as np
                try:
                    import plotly.express as px
                except ImportError:
                    st.error("Le module plotly n'est pas installé. Installez-le avec 'pip install plotly' pour voir le graphique interactif.")
                    continue
                if len(timeline) > 1:
                    t_min_min = int(np.floor(min(times)*60))
                    t_max_min = int(np.ceil(max(times)*60))
                    minutes = np.arange(t_min_min, t_max_min+1)
                    timeline_arr = np.array([(t.total_seconds()/60, v) for t,v in timeline])
                    occ_values = []
                    for m in minutes:
                        idx_ev = np.searchsorted(timeline_arr[:,0], m, side='right')-1
                        occ = timeline_arr[idx_ev,1] if idx_ev >= 0 else 0
                        occ_values.append(occ)
                    base_day = pd.Timestamp(jour)
                    dt_series = [base_day + pd.to_timedelta(m, unit='m') for m in minutes]
                    df_minutes = pd.DataFrame({
                        'DateHeure': dt_series,
                        'Occupation': occ_values
                    })
                    # Graphique en barres interactif
                    fig = px.bar(
                        df_minutes,
                        x='DateHeure',
                        y='Occupation',
                        labels={'DateHeure':'Heure','Occupation':'Nb pneus'},
                        title=f"Remplissage convoyeur {jour.strftime('%d/%m/%Y')}"
                    )
                    fig.update_layout(
                        xaxis=dict(showgrid=False),
                        yaxis=dict(range=[0, total_positions+1]),
                        hovermode='x unified'
                    )
                    fig.update_traces(hovertemplate='<b>%{x|%H:%M}</b><br>Occupation: %{y} pneus<extra></extra>')
                    st.plotly_chart(fig, use_container_width=True, config={'displaylogo': False, 'modeBarButtonsToRemove': ['select2d','lasso2d']})

                    # Synthèse par heure : moyenne d'occupation
                    df_minutes['Heure'] = df_minutes['DateHeure'].dt.hour
                    synth_tab = df_minutes.groupby('Heure')['Occupation'].mean().round(2).reset_index()
                    synth_tab['Heure'] = synth_tab['Heure'].apply(lambda h: f"{h:02d}:00")
                    synth_tab = synth_tab.rename(columns={'Occupation':'Nb moyen pneus'})
                    st.write("Synthèse par heure : nombre moyen de pneus dans le convoyeur")
                    st.dataframe(synth_tab, hide_index=True)
                else:
                    st.info("Données insuffisantes pour tracer la courbe interactive.")
