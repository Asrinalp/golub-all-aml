# Golub ALL/AML Mikrodizi Verisinde Lösemi Alt Tipi Sınıflandırması

Akut lenfoblastik lösemi (ALL) ile akut miyeloid löseminin (AML) gen ifadesi verisinden ayrılması
üzerine bir çalışma. Ana veri seti Golub ve ark. (1999); bağımsız bir pediatrik kohort olan GSE14062
dış doğrulama için kullanılır.

**Durum:** Depoda şu an veri kalitesi denetimi, keşifsel veri analizi (EDA) ve veri setleri arası
hasta örtüşmesi analizleri var. Sınıflandırma modelleri ve dış doğrulama sonraki adımdır.

## Tek komutla yeniden üretim

Python 3.11 ve 3.13 ile sınandı. Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python run_all.py
```

Linux/macOS: `.venv/bin/python` kullanın. İlk çalıştırma yaklaşık 20 dakika sürer: GSE14062 örnek
örnek alınır (140 istek) ve MILE matrisi 886 MB'tır. MILE adımını atlamak için `--skip-mile` kullanın. Sabit tohum
`SEED = 42` (`src/common.py`). Sayısal sonuçlar `results/eda_summary.json`, şekiller
`results/figures`, tablolar `results/tables` altındadır.

## Depo yapısı

| Dosya | İçerik |
|---|---|
| `src/step1_download.py` | Ham veriyi indirir, SHA-256 özetlerini `data/raw/MANIFEST.json` dosyasına yazar |
| `src/data_io.py` | Okuyucular; etiketleri hasta numarasıyla eşler, prob→gen sembolü eşlemesi |
| `src/step2_quality.py` | Eksik/INF, negatif/uç değerler, P/A/M çağrıları, etiket doğrulaması, dizi kalitesi |
| `src/step3_distributions.py` | Betimsel istatistikler, log öncesi/sonrası dağılımlar, sınıf dağılımı (Şekil 1-2) |
| `src/step4_dimred.py` | PCA, açıklanan varyans, t-SNE, PC-ortak değişken ilişkisi (Şekil 3) |
| `src/step5_genes.py` | Sınıflar arası testler, FDR, permütasyon, ısı haritası, kümeleme, GSE14062'de imza skoru (Şekil 4) |
| `src/step6_shift.py` | Eğitim-test kayması ve veri setleri arası farklar (Şekil 5-6) |
| `src/step7_overlap.py` | Golub ile GSE14062 ve Armstrong arasında hasta örtüşmesi |
| `src/step8_mile_overlap.py` | GSE14062 ve Golub ile MILE arasında hasta örtüşmesi; `results/tables/gse14062_in_mile.csv` |
| `report/literature_search/` | Literatür taraması kaydı: OpenAlex IEEE betiği ve çıktısı, elle yapılan Europe PMC ve Google Scholar aramalarının sayıları |

Ham veri depoya konmaz (`.gitignore`); `run_all.py` indirir.

## Veri kaynakları

| Veri seti | Kaynak | Kullanım koşulu |
|---|---|---|
| Golub 1999 (ana veri seti) | [Kaggle crawford/gene-expression](https://www.kaggle.com/datasets/crawford/gene-expression), [Broad Institute](https://pubs.broadinstitute.org/mpr/projects/Leukemia/Files_descriptions.txt) | Kaggle etiketi CC0; Broad sayfasında açık lisans yok; DOI 10.1126/science.286.5439.531 atfı zorunlu |
| GSE14062 (dış doğrulama) | [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE14062), E-GEOD-14062; DOI 10.1186/1755-8794-2-36 | [GEO feragatnamesi](https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html); makale CC BY |
| MILE GSE13159 (üçüncü veri seti adayı; şimdilik yalnızca örtüşme denetimi) | [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159), E-GEOD-13159; DOI 10.1200/JCO.2009.23.4732 | GEO feragatnamesi |
| Armstrong 2002 (elenen aday; yalnızca örtüşme denetimi) | [Broad Institute](https://pubs.broadinstitute.org/mpr/projects/MLL/expression_data.txt); GEO/ArrayExpress kaydı bulunamadı | Hak sahibinden açık lisans bulunamadı; DOI 10.1038/ng765 atfı zorunlu |
| Platform anotasyonu | NCBI GEO GPL80, GPL91, GPL570 | [GEO feragatnamesi](https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html) |

## Sızıntı önlemleri

- Golub'un gen sıralaması ve filtresi `step5_genes.py` içinde **yalnızca 38 eğitim örneğiyle** yapılır.
- 34 test örneğinin etiketleri yalnızca betimsel amaçla kullanılır: sınıf dağılımı, etiket doğrulaması
  ve eğitim-test kaymasının sınıf bileşiminden ayrılması. Hiçbir gen listesi veya model kararı buradan çıkmaz.
- MILE kullanılırsa GSE14062 ile eşleşen örnekler (`results/tables/gse14062_in_mile.csv`) çıkarılmalıdır.
- EDA'daki filtre ve ölçekleme kararları modele taşınmaz; modelleme aşamasında aynı adımlar çapraz
  doğrulama katlarının içinde yeniden uygulanacaktır.

## GenAI beyanı

Bu çalışmada üretken yapay zekâ (GenAI) araçlarından yararlanılmıştır; kullanımın ayrıntılı beyanı
proje raporlarıyla birlikte sunulmaktadır.
