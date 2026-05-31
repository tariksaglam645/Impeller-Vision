# ImpellerVision — Üretime Taşıma: Donanım Brief'i (Aremak)

**Amaç:** Bu demodaki yazılım hattını (MobileNetV2 + Grad-CAM) gerçek bir döküm
çark hattında çalışan bir **inline kalite kontrol istasyonuna** dönüştürmek. Aşağıda
kamera / lens / aydınlatma / mekanik seçimi gerçek üretim koşullarına göre özetlenmiştir.

---

## 1. Görüntüleme zinciri kararları

### Kamera (area-scan, makine görüşü)
- **Sensör tipi:** Global shutter zorunlu — çark konveyörde hareket halindeyse rolling
  shutter çarpıtma yapar. Sabit istasyonda (pick-and-place) rolling shutter de olur.
- **Çözünürlük hesabı:** Tespit edilmesi gereken en küçük kusur ~0,3 mm, çark çapı
  ~150 mm ise; kusuru güvenle ayırt etmek için defekt başına en az **3–4 piksel** isteriz.
  → 150 mm / (0,3 mm / 4 px) ≈ **2000 px** kenar. Pratikte **5 MP (2448×2048)** bir
  sensör güvenli pay bırakır.
- **Arayüz:** GigE Vision (kablo mesafesi/maliyet dengesi) veya yüksek hız için USB3.
  Endüstride **Basler ace / FLIR Blackfly** sınıfı yaygın ve sürücü desteği iyi.
- **Renk vs mono:** Kusur geometriktir, renk bilgisi gereksiz → **monokrom** sensör
  daha yüksek gerçek çözünürlük ve ışık verimi verir.

### Lens
- **Odak uzaklığı:** Çalışma mesafesi (WD) ve görüş alanına (FOV) göre seçilir.
  WD ≈ 300 mm, FOV ≈ 180 mm, 2/3" sensör için **f ≈ 16–25 mm** C-mount lens.
- **Ölçüm/konum kritikse telesentrik lens** değerlendirilmeli (perspektif hatası ~0),
  ancak yalnız yüzey kusuru sınıflandırması için standart fixed-focal yeterli.
- Düşük distorsiyonlu, makine görüşü sınıfı (megapiksel uyumlu) lens seçilmeli.

### Aydınlatma — **projenin en kritik kararı**
Döküm metal yüzey hem mat hem yer yer parlaktır; yanlış ışık kusuru gizler ya da sahte
gölge üretir. Kusur tipine göre:
- **Çizik / çatlak / yüzey gözenekleri:** **düşük açılı (low-angle / dark-field)**
  halka aydınlatma — yüzeydeki girinti-çıkıntılar gölge yaparak öne çıkar.
- **Parlak/işlenmiş yüzeyde yansıma sorunu:** **dome (diffüz kubbe)** veya
  **koaksiyel** aydınlatma — speküler parlamayı bastırır, homojen aydınlatır.
- **Renk:** monokrom sensörle **kırmızı/IR** LED iyi kontrast ve ortam ışığına bağışıklık verir.
- **Strobe + tetikleme:** hareketli parçada LED'i kısa darbeyle (strobe) kamerayla
  senkronize edip hareket bulanıklığını dondurun.
- **Mahfaza:** İstasyonu kapatıp ortam ışığını dışarıda bırakın — tekrarlanabilirlik için şart.

> **Demodaki "Aydınlatma Etkisi" toggle'ı tam bu noktayı gösterir:** parlaklık/kontrast
> bozulunca modelin güveni düşer. Üretimde sabit, doğru seçilmiş aydınlatma = yüksek doğruluk.

### Mekanik & tetikleme
- Çarkın **tüm yüzeyini** görmek için döner tabla (her N° bir kare) ya da çok kameralı
  açı dizisi. Tek kare yalnız bir yüzü kapsar (demodaki `*_front` gibi).
- Parça sensörü/enkoder ile **hardware trigger** → her parça aynı pozisyonda çekilir.
- Fikstür ile konum tekrarlanabilirliği (±1 mm) sağlanır.

### Hesaplama (edge)
- Hat hızı düşük-orta ise **endüstriyel PC + GPU** ya da **NVIDIA Jetson Orin** üzerinde
  bu model rahat 30+ FPS koşar (MobileNetV2 hafif). TensorRT/ONNX ile optimize edilir.

---

## 2. Yazılım tarafında üretime geçiş

1. **Gerçek hat verisiyle yeniden eğitim:** Demo Kaggle döküm verisiyle eğitildi.
   Aremak hattının kendi kamerası/ışığı/çarkıyla toplanan görüntülerle **fine-tune**
   şart (domain shift). İlk haftalarda insan onayı + veri toplama döngüsü.
2. **Lokalizasyonun güçlendirilmesi:** Şu an kutu/maske etiketi olmadığı için
   **Grad-CAM (zayıf denetimli)** kullanıldı — kusur *yerini yaklaşık* gösterir.
   Etiketli veri biriktikçe **segmentasyon (U-Net) veya YOLO** ile piksel/kutu seviyesi
   lokalizasyona yükseltilir.
3. **Karar eşiği:** PASS/FAIL eşiği (şu an 0.5) müşterinin **yanlış-negatif maliyetine**
   göre ayarlanır — kaçan kusur pahalıysa eşik kusur lehine kaydırılır (recall öncelikli).
4. **İzlenebilirlik & MLOps:** Her karar görüntü + skor + ısı haritasıyla loglanır;
   model versiyonlama, drift izleme, periyodik yeniden eğitim.
5. **Entegrasyon:** PASS/FAIL sinyali PLC'ye (dijital çıkış / OPC-UA) bağlanıp
   reddetme mekanizmasını (pusher/robot) tetikler.

---

**Özet:** Demo, yazılım hattının çalıştığını kanıtlıyor. Üretim doğruluğunu belirleyen
asıl faktör **görüntüleme zinciri** (özellikle aydınlatma) ve **hat verisiyle yeniden
eğitimdir**. Aremak'ın uzmanlık alanı olan kamera/lens/aydınlatma mühendisliği, bu
sistemin başarısının %50'sidir — yazılım diğer yarısı.
