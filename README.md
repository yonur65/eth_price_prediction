# eth_price_prediction
 Yapay Sinir Ağları ile Etherium Fiyat Tahmin Analizinde PSO ve Genetik Algoritmaların Optimizasyonu
 
Yakup Önür, Özgür Ağrali, Gürcan Çetin

# Özet
İnsan yaşamındaki dijitalleşmenin artması ile beraber, kullanılan araç ve gereçlerinde giderek daha fazla dijitalleştiği bilinmektedir. Bu durumda insanlığın en bütünleyici ve yaşamsal faaliyetleri sürdürücü gücü olarak paranın da dijitalleşmesi her geçen gün artmaktadır. Günümüz dünyasında dijital para denilen kripto paraların kullanımı oldukça artmaktadır. Geçmiş bilgilerden, bugüne ya da geleceğe yönelik bir bilgi elde etme isteği insanda hep vardır. Geleceğe yönelik tahmin çalışmaları için en güncel çalışmalar Yapay Zeka ve onun alt dallarından olan Yapay Sinir Ağları(YSA) modelleri ile yapılmaktadır. YSA modellerinin öğrenme parametreleri olan ağırlık matrisi(w) ve bias vektörü(b) değerlerinin minimum hata yani maliyeti veren değerler optimum değerler olarak seçilir. YSA Geri yayılımlı öğrenme algoritması olarak; Parçacık Sürü Algoritması(PSO), Genetik Algoritma(GA) ve Gradyan İniş Yöntemi(GİY) optimizasyon algoritmaları üzerinde ayrı ayrı işlemler gerçekleştirilmiştir. Ayrıca her bir optimizasyon algoritması farklı değerleri optimize değerler olarak bulduğu için bu çalışmada, çalışılan veri setinde gerçek değere en yakın değerleri veren algoritmayı bulmayı hedeflemektedir.
Anahtar Sözcükler: Yapay Sinir Ağları, PSO, Genetik Algoritma, Etherium, Zaman Serisi


 # 1.Giriş
Etherium günümüzde piyasa değeri yüksek olan sıralamaya göre ikinci sırada yer alan önemli bir kripto para birimidir[1]. Kripto paraların değerlerinin normal borsa değerlerinden daha hareketli ve daha esnek olduğu bilinmektedir. Etherium’un üretildiği günden bugüne kadarki, dolar endeksli piyasa kapanış değerleri zaman analizi problemi olarak ele alınıp fiyat tahmini için YSA kullanılmıştır. Çok yoğun bir şekilde iniş ve kalkış gösteren bu paranın kapanış değerlerinin gelecek değer tahmini için Gradyan İnişi geri yayılımlı optimizasyon algoritması olan standart YSA modeli kullanmak yeterli olmayabilir. Bu soruna cevap olabilmek adına aynı veri seti üzerinde hem Gradyan İniş Yöntemi(GİY), hem PSO hem de GA optimizasyon algoritmalarını kullanarak yapılan tahminlerin hata değerleri incelenmiştir.
Bu çalışmada[2] en yüksek piyasa değeri olan Bitcoin kripto parasının gelecekteki değerlerin tahmin etmek için YSA modelindeki parametrelerin eğitimi için GA kullanılmış ve bu şekilde daha otomatik ve etkili bir sonuç alındığı vurgulanmıştır.
Benzer bir çalışmada[3] vücut-yağ oranı verileri üzerinde regresyon tahmini için YSA modelinin GİY ve PSO algoritmaları ile eğitim sonuçları incelenmiş ve bu inceleme sonucunda GİY’in çok daha hızlı ve başarılı bir tahmin yaptığı görülmektedir. Ayrıca her iki algoritma için 200 iterasyondan fazlasının çok fazla etkili olmadığı vurgulanmıştır.
Bu çalışma[4] meme kanseri teşhisi üzerine bir sınıflandırma problemini çözmeye yönelik yapılmıştır. Sınıflandırma tahmini için YSA modeli üzerinden GİY ve PSO geri yayılım algoritmaları kullanılmış olup, sınıflandırma problemlerinde PSO’nun çok daha hızlı ve başarım performansı yüksek bir sonuç çıkardığı sonucuna varılmıştır. Ayrıca PSO parçacık sayısının kayda değer bir değişim yaratmadığı da görülmektedir.


# 2.Materyal ve Metot
Şekil 1’de belirtildiği üzere çalışmada; veri seti açıklanması ve ön işlemesi, YSA modelinin
GİY, PSO ve GA ile hata minimizasyonu sonucunu bulmaya dair ağırlık ve bias optimizasyon çalışması yapılacaktır.

## 2.1. Veri Seti
Kullanılan veri seti, Etherium kripto para biriminin geçmişe yönelik fiyatlarının yer aldığı “csv” formatındaki veri setidir[1]. Etherium’un dolar endeksli günlük değerlerinin olduğu veri seti, 07/08/2015 ile 25/04/2021 tarihleri arasındaki fiyatlardan oluşmaktadır. Toplamda 2089 güne ait değerler bulunmaktadır. Tarihsel olarak geçmişten geleceğe doğru bir kronolojik sıralama içerisindedir. 2089 gün için 7 farklı özellik içermektedir. Bu özellikler Şekil 2’de görüldüğü gibi; tarih, açılış değeri, en yüksek, en düşük,kapanış değeri, en yakın değer ve hacim değişkenlerinden oluşmaktadır.

## 2.2. Yapay sinir ağları
Yapay sinir ağları, insan beyninin ve sinir sisteminin öğrenme sisteminin matematiksel olarak modellendiği bir yapı olduğu bilinmektedir.[5] YSA, özellikle istatistiksel tabanlı makine öğrenmesi tekniklerinden farklı olarak insan öğrenme biçimi taklit edilerek geliştirilmiş dinamik bir modeldir. Bu öğrenme insan etki-tepki sistemi düşünülerek tasarlanmıştır. İnsan organlarının algıladığı bir girdi bilgisinin, anında işlenip bir çıktı değer üretmesi ile gerçek zamanlı uygulamalara uyarlanmıştır.
YSA tek katmanlı ve çok katmanlı perseptron olarak ikiye ayrılır. Tek katmanlı perseptron daha çok doğrusal problemlerin çözümünde hızlı ve etkili çözüm geliştirirken, çok katmanlı perseptronlar daha karmaşık sistemlerde başar göstermektedir. Model yapısı gereği girdi ve çıktı arasında bu katmanlar yer alır. Her bir katmanda ise bir ya da birden çok nöron bulunmaktadır. Her bir nöron da girdi ve çıktı değer alarak geri ve ileri yayılımlı işlem yapabilir durumdadır. 2 veya daha fazla katman olan YSA modellerine Derin Sinir Ağlar denilmektedir.

Şekil 3’de görüldüğü üzere tek katmanlı YSA modellerindeki matematiksel işlemlerin girdiler-çıktılar arasında gerçekleşmektedir. Her bir nörona girdi sayısı kadar ağırlık değeri ve bir tane de bias değeri bulunmaktadır. Ağırlık ve bias değerleri matris ya da vektör şeklinde işlenmektedir. Girdi verilerini bir nörona aktarılırken üzerinde bir ağırlık(weight) değeri ile çarpılarak gönderilir. Bulunan bu çarpım değerleri ve bias değeri toplanarak bulunan değer nöronun çıktı değeri hesaplanmaktadır. Bu şekilde işlemler her bir katman ve nöronların yapısına göre tekrar eder. Ayrıca çıktı değerlerin model çıktısına uygun duruma getirilmesi için aktivasyon fonksiyonları da kullanılmaktadır.

## 2.3. Gradyan İniş Yöntemi(GİY)
YSA modellerinde ileri beslemeli ağlar için tam bir öğrenme gerçekleşmediği için geriye yayılım algoritmaları kullanılmaktadır. Geri yayılım algoritmaları model parametrelerinin güçlenmesini sağlar. Bu çalışmada,[6] YSA’da kullanılan en yaygın ve etkili geri yayılım algoritması GİY’in türev tabanlı matematiksel bir hesaplama yapıldığı açıklanmıştır. Başlangıç olarak rastgele bias ve ağırlık değerleri verilmektedir. Daha sonra bu parametrelerden elde edilen çıktı kaydedildikten sonra tekrar en başa bu değerler güncellenerek işlemler tekrarlanır. Daha sonra bu çıktıların gerçek değer ile arasındaki farka bakılır buna göre minimum hata değerini yakalayan parametre değerleri optimize edilmiş değerlerdir.


GİY algoritması maliyet-hata fonksiyonunu en aza indirgeyerek en düşük hatayı bulmak için kullanılmaktadır. Şekil 4’e göre GİY’in çalışma şekli şu şekildedir :
1. Bir J(W,b) başlangıç noktasıdır..
2. J’nin değerlerini azaltmak için uzayda sürekli olarak küçük adımlar atılır. Her
iterasyonda gradyanlara bağlı ağırlık ve biaslarda güncelleme yapılır.
3. J>=0 olacağı için J’nin değerleri sürekli azaltılır.
4. Bu işlemler sonsuza kadar devam edemeyeceği için bir noktada durur.
GİY algoritmasının çalışma düzeni yukarıdaki maddelerde belirtildiği gibi olduğuna
göre, her defasında bulunan konumun o anki minimum ya da maksimum değer değil, bütün noktalar arasındaki maksimum ya da minimum noktada durmayı hedeflemelidir. Global minimum ya da maksimum noktayı yakalayıp orda durabilen sonuçlarda en iyi sonucu vermektedir.

## 2.4. Parçacık Sürü Algoritması
Parçacık sürü optimizasyonu, 1995 yılında Dr.Eberhart ve Dr.Kennedy tarafından geliştirilmiş olup popülasyon temelli sezgisel bir optimizasyon tekniğidir[7]. Genel olarak her parçacığa rastgele konum atanarak her iterasyonda konumları değiştirilerek en iyi sonuç bulunmaya çalışılır. Sürüdeki en iyi konuma global olarak belirlenip, diğer parçaların buna yaklaşması sağlanır.

## 2.5. Genetik Algoritma
Genetik algoritma (GA) , permütasyon tabanlı bir optimizasyon yöntemidir. Olasılıklar üzerinde yakınsama yaparak arama yapar[8]. Doğadaki evrimsel sürece benzer bir şekilde çalışır. Algoritmanın girdilerinden olan uygunluk fonksiyonu (fitness) tanımlanır. Geçiş ve Mutasyon gibi genetik operatörler, evrim sürecinin birçok aşamasında rassal olarak uygulanmaktadır, bu yüzden gerçekleşme olasılıkları belirlenmelidir. Yakınsama kriterleri sağlanıp ve optimal maliyet ile problem çözülür.


# 3. Zaman Serisi ile Etherium Fiyat Tahmininin Yapılması
Zamanla değişen verilerin analizi ve tahmini üzerine birçok çalışma yapılmıştır. Ayrıca zaman serisi üzerine yapılan bir çok çalışmanın da ekonomik çalışmalar doğduğu da bilinmektedir.[9,10] Bu çalışmada YSA modellerinin GİY, GA ve PSO algoritmaları ile parametre optimizasyonu işlemlerinin karşılaştırılmalı olarak başarımlarının anlaşılması için; üç farklı yöntem için kabul edilen sabit bazı değerlerin olması gerekmektedir. Her üç algoritma için YSA modellerinde kabul görülen ön değerlerler şunlardır:
İterasyon Sayısı = 200 Katman Sayısı = 1 Nöron Sayısı = 7 Gündelik periyot = 7 Eğitim Verisi = %80
* Öğrenme/parametre güncelleme ve çalışma sayısı # YSA modelindeki katman sayısı
* Her bir katmanda yer alan nöron sayısı
* Zaman serisi için her 7 günde bir 8. gün tahmini
* Test verisi de veri setindeki son %20’lik kısım

 Bu çalışmanın amacı yukarıda belirtilen YSA modellerin sabit kabul değerlerine göre her bir algoritmanın eğitim ve test başarım ölçütleri karşılaştırılacaktır. Bunları yaparken YSA modelin içerisindeki ağırlık ve bias parametrelerin değerlerine göre elde edilen tahmin değerinin gerçek değer ile karşılaştırılması yapılmaktadır. Hata fonksiyonu(gerçek değer-tahmin değeri) ile ölçülen hata değerlerin toplam olarak en küçük değeri veren algoritma belirlenir. Her bir algoritma için YSA modelinden çıkan sonuçlar için hata değerlerinin minimize edilmesi amaçlanmaktadır.
Tahmin edilen değerlerin, zamana dayalı değişen değerler olması sebebiyle zaman serisi yöntemleri ve analizi de gerekmektedir. Bu doğrultuda elde edilen veri setine göre farklı analizler ve analizler yapılmaktadır. Çalışmada kullanılan verilerin günlük değerler olması ve hareketli veriler olduğu için bir haftalık süre ile tahmin analizi gerçekleştirmek gerekmektedir.


## 3.1. Veri Seti Ön İşleme
Geçmiş günlerdeki kapanış fiyatlarına bakarak gelecek günlerdeki kapanış fiyatının tahminin yapılabilmesi için, çok iyi tahmin yapabilen bir modele ihtiyaç vardır. Şekil 7’de görüldüğü üzere geçmişten günümüze kadar, günden güne artış/azalış kapanış değerlerinin çok esnek olduğu açıktır. Bu sebeple, bu tür veri setlerine yönelik doğru tahmin yapmak oldukça güçtür.
Elde edilen ham veri seti üzerine bir YSA modeli kurmak bir çok yanlı ve yanlış sonuçlara yol açacağı bilinmektedir. Veri seti içerisindeki değişken tipleri, eksik ya da aykırı verilerin varlığı, değişken değerlerin normalizasyonu vb. gibi işlemler uygulanacak yönteme göre düzenlenmelidir. Veri seti kronolojik bir sıralama halinde olduğu için sıralama üzerine bir işlem yapılması gerekmemektedir. Ayrıca günlük kapanış değerleri öğrenme, ve tahmin etme işlemi gerçekleştirileceği için sadece kapanış değeri yani “Close” değişkeni üzerine işlemler gerçekleştirilecektir.

İlk olarak veri seti içerisindeki kapanış değişkeninine ait eksik veri taraması yapıldı ve 4 günün değerleri eksik olduğu tespit edilmiştir. Daha sonra bu bazı görsel veri analizleri sonucunda bu eksik verilerin nasıl giderileceği üzerine çıkarımlar yapılmıştır. Veri seti günlük borsa değerlerin iniş çıkış değerleri olması sebebiyle, o günün kapanış değeri eksik değerleri önceki ve sonraki gün değerlerinin ortalaması ile giderilmiştir.
Kullanılan veri setinde yüksek ve düşük değerler arasında matematiksel olarak çok büyük fark olduğu gözlemlenmiştir. YSA modelinin değişkenler değerleri arasındaki değerlerin matematiksel olarak büyük-küçük değerlerden etkilenip yanlış bir sonuç elde etmemesi için, değerler üzerine normalizasyon işlemi yapılmıştır. Bu işlem “mapminmax” diye tanımlanan fonksiyonel bir işlemdir. Seri içerisindeki en büyük değere 1 ve en küçük değere 0 değerini vererek arada kalan değerlerin 0-1 ölçüm aralığına göre ölçekleme yapmaktadır. Fakat test veri seti çıktısını daha önce normalize edilen ölçeğe göre tekrardan denormalize işlemi yapılmıştır. Gerçek para değeri ile tahmin edilen değerin normal ölçek ile değerlendirilmesi, test işleminde daha net olarak görülmektedir. Bu durum özellikle YSA modellerinin başarımını çok etkilemektedir.
Ayrıca YSA modelinin öğrendiği ve hiç rast gelmediği gün verileri için tahminlerinin analizi gerekmektedir. Bu durum veri setini eğitim ve test verisi olarak iki farklı veri seti olarak ayrılması demektir. Verilerin ilk %80’lik kısmını eğitim, geri kalan son %20’lik kısmı ise test verisi olarak kullanılmıştır.
Son olarak, günlük kapanış değerleri(Close) değişkeninin zaman serisi olarak tahmin edilmesi için tek sütun halindeki değişkenin bu yapıya uygun bir biçimde düzeltilmesi gerekmektedir. Bu doğrultuda, çalışmada haftalık bazlı tahmin işlemi hedeflendiği için, her 7 günde bir 8. günün kapanış değeri olacak şeklinde bir düzenleme yapılmaktadır. Tek boyutlu veriler artık 8 boyutlu bir görsel şeklinde temsil ediyor oldu. Her bir satır için bağımlı değişken son sütun değişkeni(8. günün kapanış değeri), bağımsız değişkenler için ondan önceki 7 gün sütunları(değişkenleri) olacak şeklinde düzenlenmiştir.

## 3.2. Gradyan İniş Yöntemi ile Tahminin Yapılması
 Şekil 8. GİY ile optimize edilmiş YSA’nın tahmin sonuçları GİY, YSA alanında popüler bir yöntemdir. Ama, eğitim verisi bakılarak, en yüksek doğruluğu bulmak ya da hata oranını en aza indirmektir. İleri besleme tek katmanlı ve 7 nörondan oluşan bir YSA modeli oluşturulup, 200 iterasyonda veri seti üzerine eğitim ve test işlemleri yapılmıştır. GİY algoritmasının diğer parametre değerleri varsayılan değerler olarak kabul edilmiştir. Bu işlemlerin sonucunda Şekil 8’de görüldüğü üzere eğitim verisini çok iyi öğrenen model, test verisi tahmininde çok da başarılı bir tahmin yapılmadığı görülmektedir. Test verisi tahmininde hata değerlerinin de son verilere doğru gitgide arttığı görülmektedir.

## 3.3. Parçacık Sürü Algoritması ile Tahminin Yapılması
 Şekil 9. PSO ile optimize edilmiş YSA’nın tahmin sonuçları
PSO bir çok alanda kullanılan optimizasyonlardan yöntemlerinden biri olup gayet iyi sonuçlar vermekte. GİY’ye göre yavaş ve GA’ya göre çok hızlı bir CPU zamanına sahip. Bu çalışmada ortak olarak 200 iterasyon kullanıldığından hızlı bir sonuç alınması için sürü 10 olarak belirlendi. Diğer sabit değerler üzerinde denemeler yapılarak en iyi çözümün sabit değerler kullanılmasına karar verildi. Şekil 9’da görüleceği üzere GİY’den daha iyi bir sonuç elde edilmiştir.

## 3.4. Genetik Algoritma ile Tahminin Yapılması
GA çok uzun yıllardır kullanılan algoritmalar arasındadır. Şekil 10’da görüleceği üzere çalışmada en iyi sonuca GA ile ulaşılmıştır. Varsayılan değerleri ile gen limiti 100 olarak belirlenip diğer optimizasyonlarda kullanıldığı gibi 200 iterasyonda kullanıldı.

# 4.Tartışma
İleri beslemeli geri yayılımlı YSA modelleri için yaygın olarak GİY algoritması ile optimizasyon ve öğrenme algoritması kullanılmasına rağmen bu çalışmadaki sezgisel algortimların da YSA modellerin parametre değerlerin optimize edilip minimum hata değeri bulunması için kullanılabileceği görülmektedir. Farklı veri setlerinde, farklı iterasyon sayılarında, farklı katman ve nöron sayılarındaki modeller için farklı optimizasyon algoritması uygun olabilmektedir.
Yapılan bu çalışmada her üç optimizasyon algoritması için eğitim verisinin öğrenme oranı yüksek ve hata oranı düşük olmasına rağmen test eğitim serilerinin tahmininde daha büyük hatalar gözlemlenmiştir. Özellikle son yıllarda gittikçe daha çok artan kripto para piyasasının Etherium üzerindeki etkilerine de bakıldığında,modelin son zamanlardaki ani artışları tahmin etmekte zorlanmıştır. Bu eğitim ve test verilerin büyüklüklerinin oranları değiştirilerek en optimum %80 eğitim ve %20 test verisi tespit edilmiştir.
İterasyon sayısının 200’den fazla olduğu durumlarda sırasıyla GA olmak üzere PSO ve GİY yöntemlerinin çalışma zamanını oldukça yavaşlatmaktadır. Bu sebeple özellikle GİY için 200’den fazla iterasyon ile daha iyi ve daha hızlı sonuç alınacağı belirlenmiştir. Fakat GA ve PSO için aynı durum geçerli değildir. Belli bir iterasyon sayısından sonra çok fazla bir değişim gözlemlenmemiştir.
YSA katman sayısının fazla olması öğrenme sürecini yavaşlattığı bilinmektedir. Bu çalışmada katman sayısı bir olarak alınmıştır. Tek katmanlı YSA modelindeki geri yayılımlı algoritmalardan en iyi başarımı GA vermektedir. PSO da GA gibi daha yakın sonuç çıkarırken. GİY tek katmanlı bir yapı için doğru tahmin başarımı düşük olmuştur.

# 5.Sonuç
Bu çalışma sürecinde YSA ile zaman serisi tahminlerde optimizasyonun çok büyük bir önem arz ettiği gözlemlenmiştir. Yapılan optimizasyonlarda en iyi sonucu GA ile alınmış olmasının en büyük nedeni CPU zamanı en çok harcayan optimizasyon olmasından kaynaklıdır. Çalışmanın ileriki aşamalarında diğer kripto paralar üzerinde çalışılarak genel bir model ve optimizasyon senaryosu ortaya çıkarılacak. Herkesin buna ulaşabilmesi için bir web site arayüzü geliştirilerek kullanıma açılması planlanmaktadır.

# Kaynakça
[1] Yahoo Finance [Ethereum USD (ETH-USD)] Erişim adresi : https://finance.yahoo.com/quote/ETH-USD/history/

[2] Nayak, S. C. (2021). Bitcoin closing price movement prediction with optimal functional link neural networks. Evolutionary Intelligence, 1-15.

[3] ŞİMŞEK, Ö. İ., ALAGÖZ, B. B., & KARCİ, A. Parçacık Sürüsü Optimizasyonun Yapay Sinir Ağının Eğitiminde Uygulanması ve Ağırlık Entropi Değişiminin İncelenmesi. Bilgisayar Bilimleri, 5(2), 126-136.

[4] Çevik, K. K., & Koçer, H. E. (2013). Parçacık sürü optimizasyonu ile yapay sinir ağları eğitimine dayalı bir esnek hesaplama uygulaması. Süleyman Demirel Üniversitesi Fen Bilimleri Enstitüsü Dergisi, 17(2), 39-45.

[5] Yegnanarayana, B. (2009). Artificial neural networks. PHI Learning Pvt. Ltd..

[6] Ruder, S. (2016). An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747.

[7] Kennedy, J., & Eberhart, R. (1995, November). Particle swarm optimization. In Proceedings of ICNN'95-international conference on neural networks (Vol. 4, pp. 1942-1948). IEEE.

[8] Popov, A. (2005). Genetic algorithms for optimization. User Manual, Hamburg, 2013. [9] Kaastra, I., & Boyd, M. (1996). Designing a neural network for forecasting financial and economic time series. Neurocomputing, 10(3), 215-236.

[10] Ghazali, R. (2007). Higher order neural networks for financial time series prediction. Liverpool John Moores University (United Kingdom).

