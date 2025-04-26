# Agrotopya Yapay Zeka Modeli Geliştirme Raporu

## Özet

Bu rapor, TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projeleri Desteği Programı kapsamında geliştirilen "Toprak ve Sıcaklık Sensörleri Kullanarak Topraktaki Ürünlerin Su İhtiyacını Agrotopya ile Tespit Etmek" projesinin yapay zeka modeli geliştirme sürecini detaylandırmaktadır. Proje, domates yetiştiriciliğinde toprak nemi, sıcaklık ve hava nemi sensörlerinden gelen verileri kullanarak sulama ihtiyacını belirleyen bir yapay zeka modeli ve bu modeli kullanan bir API geliştirmeyi amaçlamaktadır.

Geliştirilen yapay zeka modeli (Agrotopya), sensör verilerini analiz ederek sulama pompasının açılıp kapatılması gerektiğini yüksek doğrulukla tahmin edebilmektedir. Model, Flask tabanlı bir API ile entegre edilmiş ve mobil uygulama ile iletişim kurabilecek şekilde tasarlanmıştır.

## İçindekiler

1. [Giriş](#giriş)
2. [Veri Analizi ve İşleme](#veri-analizi-ve-i̇şleme)
3. [Yapay Zeka Modeli Geliştirme](#yapay-zeka-modeli-geliştirme)
4. [API Geliştirme](#api-geliştirme)
5. [Test ve Değerlendirme](#test-ve-değerlendirme)
6. [Sonuç ve Öneriler](#sonuç-ve-öneriler)

## Giriş

Tarımda su kaynaklarının verimli kullanımı, sürdürülebilir tarım için kritik öneme sahiptir. Bu proje, domates yetiştiriciliğinde toprak nemi, sıcaklık ve hava nemi sensörlerinden gelen verileri kullanarak sulama ihtiyacını belirleyen bir yapay zeka modeli geliştirmeyi amaçlamaktadır.

Proje kapsamında geliştirilen Agrotopya yapay zeka modeli, sensör verilerini analiz ederek sulama pompasının açılıp kapatılması gerektiğini tahmin etmektedir. Model, Flask tabanlı bir API ile entegre edilmiş ve mobil uygulama ile iletişim kurabilecek şekilde tasarlanmıştır.

### Proje Hedefleri

1. Toprak ve sıcaklık sensörlerinden gelen verileri analiz etmek
2. Sulama ihtiyacını tahmin eden bir yapay zeka modeli geliştirmek
3. Modeli bir API ile entegre etmek
4. Mobil uygulama ile iletişim kurabilecek bir sistem oluşturmak
5. Su kaynaklarının verimli kullanımını sağlamak

## Veri Analizi ve İşleme

### Veri Seti Özellikleri

Projede kullanılan veri seti, domates yetiştiriciliğinde kullanılan sensörlerden toplanan verileri içermektedir. Veri seti, 3000 adet kayıttan oluşmakta ve aşağıdaki özellikleri içermektedir:

- **Soil Moisture (Toprak Nemi)**: 300-1000 arasında değişen değerler
- **Temperature (Sıcaklık)**: 18-39°C arasında değişen değerler
- **Air Humidity (Hava Nemi)**: 38-81% arasında değişen değerler
- **Pump Data (Pompa Verisi)**: 0 (kapalı) veya 1 (açık) değerleri

Veri setinin temel istatistikleri aşağıdaki gibidir:

| Özellik | Ortalama | Standart Sapma | Minimum | Maksimum |
|---------|----------|----------------|---------|----------|
| Soil Moisture | 662.42 | 187.94 | 314.51 | 984.83 |
| Temperature | 28.44 | 6.02 | 18.00 | 38.99 |
| Air Humidity | 59.39 | 12.43 | 38.00 | 81.27 |
| Pump Data | 0.52 | 0.50 | 0.00 | 1.00 |

### Veri Ön İşleme

Veri seti üzerinde aşağıdaki ön işleme adımları gerçekleştirilmiştir:

1. **Eksik Değer Kontrolü**: Veri setinde eksik değer bulunmamaktadır.
2. **Aykırı Değer Analizi**: IQR (Interquartile Range) yöntemi kullanılarak aykırı değerler tespit edilmiş, ancak anlamlı aykırı değerler bulunmamıştır.
3. **Korelasyon Analizi**: Özellikler arasındaki ilişkiler incelenmiştir. Toprak nemi ile pompa verisi arasında güçlü bir negatif korelasyon (-0.85) bulunmuştur, bu da toprak nemi düştükçe pompanın açılma olasılığının arttığını göstermektedir.
4. **Veri Normalizasyonu**: Özellikler arasındaki ölçek farklılıklarını gidermek için StandardScaler kullanılarak veriler normalize edilmiştir.
5. **Eğitim ve Test Setlerine Ayırma**: Veri seti, %80 eğitim ve %20 test olacak şekilde bölünmüştür.

## Yapay Zeka Modeli Geliştirme

### Model Seçimi ve Karşılaştırma

Sulama ihtiyacını tahmin etmek için çeşitli makine öğrenmesi modelleri denenmiş ve karşılaştırılmıştır:

| Model | Doğruluk | Kesinlik | Duyarlılık | F1 Skoru |
|-------|----------|----------|------------|----------|
| Lojistik Regresyon | 0.9950 | 0.9934 | 0.9967 | 0.9950 |
| Random Forest | 0.9983 | 0.9967 | 1.0000 | 0.9983 |
| Gradient Boosting | 0.9967 | 0.9967 | 0.9967 | 0.9967 |
| Support Vector Machine | 0.9933 | 0.9901 | 0.9967 | 0.9934 |
| Neural Network | 0.9950 | 0.9934 | 0.9967 | 0.9950 |

Karşılaştırma sonucunda, en yüksek performansı **Random Forest** algoritması göstermiştir. Bu model, %99.83 doğruluk ve F1 skoru ile diğer modellere göre daha başarılı olmuştur.

### Model Optimizasyonu

Random Forest modeli, GridSearchCV kullanılarak hiperparametre optimizasyonu ile iyileştirilmiştir. En iyi parametreler şu şekilde belirlenmiştir:

- **n_estimators**: 50 (Ağaçtaki karar ağacı sayısı)
- **max_depth**: None (Ağaçların maksimum derinliği)
- **min_samples_split**: 2 (Bir düğümü bölmek için gereken minimum örnek sayısı)

Optimize edilmiş model, test seti üzerinde aşağıdaki performansı göstermiştir:

- **Doğruluk (Accuracy)**: 0.9983
- **Kesinlik (Precision)**: 0.9967
- **Duyarlılık (Recall)**: 1.0000
- **F1 Skoru**: 0.9983

### Özellik Önemleri

Random Forest modelinin özellik önem analizi, hangi sensör verilerinin sulama kararında daha etkili olduğunu göstermektedir:

- **Toprak Nemi**: %97.67
- **Sıcaklık**: %1.20
- **Hava Nemi**: %1.13

Bu sonuçlar, sulama kararında toprak neminin baskın faktör olduğunu göstermektedir. Bu da tarımsal açıdan beklenen bir sonuçtur ve modelimizin doğru çalıştığını doğrulamaktadır.

## API Geliştirme

### API Mimarisi

Agrotopya yapay zeka modelini kullanmak ve sensör verileri ile mobil uygulama arasında iletişim sağlamak için Flask tabanlı bir RESTful API geliştirilmiştir. API, aşağıdaki bileşenlerden oluşmaktadır:

1. **Model Entegrasyonu**: Eğitilmiş yapay zeka modeli ve veri ölçeklendirme modeli API'ye entegre edilmiştir.
2. **Endpoint'ler**: Çeşitli işlevler için farklı endpoint'ler oluşturulmuştur.
3. **Veri Saklama**: Sensör verilerinin geçmişini tutmak için basit bir bellek içi veritabanı kullanılmıştır (gerçek uygulamada kalıcı bir veritabanı kullanılacaktır).
4. **Hata Yönetimi**: API isteklerinde oluşabilecek hatalar için kapsamlı hata yönetimi mekanizmaları eklenmiştir.
5. **Loglama**: API'nin çalışması sırasında oluşan olayları kaydetmek için loglama mekanizması eklenmiştir.

### API Endpoint'leri

API, aşağıdaki endpoint'leri sunmaktadır:

| Endpoint | Metod | Açıklama |
|----------|-------|----------|
| /health | GET | API'nin sağlık durumunu kontrol eder |
| /predict | POST | Sensör verilerine göre sulama tahmini yapar |
| /history | GET | Sensör verilerinin geçmişini döndürür |
| /stats | GET | Sensör verilerinin istatistiklerini döndürür |
| /simulate | POST | Test amaçlı sensör verisi simülasyonu yapar |
| /sensor/data | POST | Raspberry Pi'dan gelen sensör verilerini alır ve işler |
| /device/command | POST | Mobil uygulamadan cihaza komut gönderir |

### Web Arayüzü

API'nin kullanımını kolaylaştırmak ve görselleştirmek için bir web arayüzü geliştirilmiştir. Bu arayüz, aşağıdaki özellikleri sunmaktadır:

1. **Sensör Verilerini Görüntüleme**: Toprak nemi, sıcaklık ve hava nemi değerlerini gerçek zamanlı olarak görüntüleme
2. **Sulama Durumu**: Mevcut sulama durumunu görüntüleme ve manuel kontrol
3. **Geçmiş Tahminler**: Son tahminleri tablo halinde görüntüleme
4. **Grafik Görselleştirme**: Sensör verilerinin zaman içindeki değişimini gösteren grafikler
5. **API Test**: API'yi test etmek için kullanıcı arayüzü

### API Dokümantasyonu

API'nin kullanımını kolaylaştırmak için kapsamlı bir dokümantasyon hazırlanmıştır. Bu dokümantasyon, aşağıdaki bilgileri içermektedir:

1. **Genel Bilgiler**: API'nin temel URL'si, kimlik doğrulama gereksinimleri ve yanıt formatı
2. **Endpoint Açıklamaları**: Her endpoint'in ayrıntılı açıklaması, parametreleri ve yanıt formatı
3. **Örnek Kullanım**: Her endpoint için örnek istek ve yanıtlar
4. **Hata Kodları**: Olası hata kodları ve açıklamaları

Ayrıca, API dokümantasyonu için Swagger entegrasyonu da sağlanmıştır.

## Test ve Değerlendirme

### Fonksiyonel Testler

API'nin doğru çalıştığını doğrulamak için kapsamlı fonksiyonel testler gerçekleştirilmiştir. Bu testler, aşağıdaki senaryoları içermektedir:

1. **Sağlık Kontrolü**: API'nin sağlık durumunun kontrol edilmesi
2. **Tahmin Testi**: Farklı sensör değerleri için tahmin yapılması ve sonuçların doğrulanması
3. **Simülasyon Testi**: Sensör verisi simülasyonunun test edilmesi
4. **Geçmiş Verileri Alma**: Geçmiş sensör verilerinin alınması ve doğrulanması
5. **İstatistik Alma**: Sensör verilerinin istatistiklerinin alınması ve doğrulanması

Tüm fonksiyonel testler başarıyla tamamlanmıştır.

### Yük Testi

API'nin yüksek yük altında performansını değerlendirmek için yük testi gerçekleştirilmiştir. Bu test, aşağıdaki sonuçları vermiştir:

- **Toplam İstek Sayısı**: 100
- **Başarılı İstek Sayısı**: 100
- **Başarı Oranı**: %100
- **Ortalama Yanıt Süresi**: 0.0220 saniye
- **Saniyedeki İstek Sayısı**: 45.35

Bu sonuçlar, API'nin yüksek yük altında bile iyi performans gösterdiğini ve gerçek dünya uygulamaları için uygun olduğunu göstermektedir.

## Sonuç ve Öneriler

### Sonuçlar

Bu proje kapsamında, domates yetiştiriciliğinde sulama ihtiyacını tahmin eden bir yapay zeka modeli ve bu modeli kullanan bir API başarıyla geliştirilmiştir. Geliştirilen sistem, aşağıdaki özelliklere sahiptir:

1. **Yüksek Doğruluk**: Model, %99.83 doğruluk ile sulama ihtiyacını tahmin edebilmektedir.
2. **Gerçek Zamanlı Tahmin**: API, sensör verilerini gerçek zamanlı olarak işleyerek anında tahmin yapabilmektedir.
3. **Kullanıcı Dostu Arayüz**: Web arayüzü, sensör verilerini ve tahminleri görselleştirerek kullanıcı deneyimini iyileştirmektedir.
4. **Kapsamlı Dokümantasyon**: API'nin kullanımını kolaylaştırmak için kapsamlı bir dokümantasyon hazırlanmıştır.
5. **Yüksek Performans**: API, yüksek yük altında bile iyi performans göstermektedir.

### Öneriler

Projenin gelecekteki gelişimi için aşağıdaki öneriler sunulmaktadır:

1. **Veritabanı Entegrasyonu**: Sensör verilerinin kalıcı olarak saklanması için bir veritabanı entegrasyonu eklenebilir.
2. **Güvenlik İyileştirmeleri**: API'ye kimlik doğrulama ve yetkilendirme mekanizmaları eklenebilir.
3. **Daha Fazla Sensör Desteği**: Toprak pH'ı, ışık yoğunluğu gibi ek sensörler entegre edilebilir.
4. **Zaman Serisi Analizi**: Sensör verilerinin zaman içindeki değişimini analiz eden modeller eklenebilir.
5. **Mobil Uygulama Geliştirme**: API ile iletişim kuran bir mobil uygulama geliştirilebilir.
6. **Çoklu Bitki Desteği**: Sistem, domates dışındaki bitkileri de destekleyecek şekilde genişletilebilir.
7. **Ölçeklenebilirlik İyileştirmeleri**: Sistem, daha büyük tarım alanlarını destekleyecek şekilde ölçeklendirilebilir.

Bu öneriler, projenin daha da geliştirilmesi ve tarımda su kaynaklarının verimli kullanımına daha fazla katkı sağlaması için yol gösterici olacaktır.
