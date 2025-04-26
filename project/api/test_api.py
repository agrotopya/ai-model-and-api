import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from datetime import datetime


# Windows uyumlu olacak
current_dir = os.getcwd()
test_dir = os.path.join(current_dir, 'api', 'test')
os.makedirs(test_dir, exist_ok=True)


# API URL
API_URL = 'http://localhost:5000'

def test_health_endpoint():
    """Sağlık kontrolü endpoint'ini test eder"""
    print("Sağlık kontrolü endpoint'i test ediliyor...")
    try:
        response = requests.get(f'{API_URL}/health')
        response.raise_for_status()
        result = response.json()
        print(f"Sağlık kontrolü başarılı: {result}")
        return True
    except Exception as e:
        print(f"Sağlık kontrolü başarısız: {str(e)}")
        return False

def test_predict_endpoint():
    """Tahmin endpoint'ini test eder"""
    print("\nTahmin endpoint'i test ediliyor...")
    
    # Test senaryoları
    test_cases = [
        {
            "name": "Düşük toprak nemi (sulama gerekli)",
            "data": {
                "soil_moisture": 400.0,
                "temperature": 28.0,
                "air_humidity": 60.0
            },
            "expected_prediction": 1  # Sulama gerekli
        },
        {
            "name": "Yüksek toprak nemi (sulama gereksiz)",
            "data": {
                "soil_moisture": 850.0,
                "temperature": 28.0,
                "air_humidity": 60.0
            },
            "expected_prediction": 0  # Sulama gereksiz
        },
        {
            "name": "Sınır değer testi",
            "data": {
                "soil_moisture": 666.0,  # Ortalama değer
                "temperature": 28.0,
                "air_humidity": 60.0
            }
            # Beklenen tahmin belirli değil, sadece API yanıtını kontrol edeceğiz
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        try:
            print(f"\nTest: {test_case['name']}")
            print(f"Gönderilen veri: {test_case['data']}")
            
            response = requests.post(
                f'{API_URL}/predict',
                json=test_case['data'],
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            result = response.json()
            
            print(f"API yanıtı: {result}")
            
            # Beklenen tahmin varsa kontrol et
            if 'expected_prediction' in test_case:
                if result['prediction'] == test_case['expected_prediction']:
                    print(f"✅ Test başarılı: Beklenen tahmin ({test_case['expected_prediction']}) ile eşleşiyor")
                else:
                    print(f"❌ Test başarısız: Beklenen tahmin ({test_case['expected_prediction']}) ile eşleşmiyor")
            
            results.append({
                "test_case": test_case,
                "result": result,
                "success": True
            })
            
        except Exception as e:
            print(f"❌ Test başarısız: {str(e)}")
            results.append({
                "test_case": test_case,
                "error": str(e),
                "success": False
            })
    
    # Sonuçları dosyaya kaydet
    with open(f'{test_dir}/predict_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return all(result['success'] for result in results)

def test_simulate_endpoint():
    """Simülasyon endpoint'ini test eder"""
    print("\nSimülasyon endpoint'i test ediliyor...")
    
    try:
        # 10 adet simüle edilmiş veri iste
        simulation_request = {
            "count": 10,
            "soil_moisture_range": [350, 950],
            "temperature_range": [20, 35],
            "air_humidity_range": [40, 80]
        }
        
        print(f"Simülasyon isteği: {simulation_request}")
        
        response = requests.post(
            f'{API_URL}/simulate',
            json=simulation_request,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        result = response.json()
        
        print(f"Simülasyon sonucu: {len(result['results'])} adet veri oluşturuldu")
        
        # Simülasyon sonuçlarını görselleştir
        if result['results']:
            soil_moisture_values = [r['soil_moisture'] for r in result['results']]
            predictions = [r['prediction'] for r in result['results']]
            
            plt.figure(figsize=(10, 6))
            plt.scatter(soil_moisture_values, predictions, c=predictions, cmap='coolwarm', s=100)
            plt.xlabel('Toprak Nemi')
            plt.ylabel('Sulama Tahmini (0: Kapalı, 1: Açık)')
            plt.title('Toprak Nemi ve Sulama Tahmini İlişkisi')
            plt.grid(True, alpha=0.3)
            plt.colorbar(label='Sulama Durumu')
            plt.tight_layout()
            plt.savefig(f'{test_dir}/simulation_results.png')
            plt.close()
            
            print(f"Simülasyon görselleştirmesi kaydedildi: {test_dir}/simulation_results.png")
        
        # Sonuçları dosyaya kaydet
        with open(f'{test_dir}/simulate_test_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        return True
    
    except Exception as e:
        print(f"Simülasyon testi başarısız: {str(e)}")
        return False

def test_history_endpoint():
    """Geçmiş endpoint'ini test eder"""
    print("\nGeçmiş endpoint'i test ediliyor...")
    
    try:
        response = requests.get(f'{API_URL}/history')
        response.raise_for_status()
        result = response.json()
        
        print(f"Geçmiş verileri alındı: {result['total_records']} kayıt")
        
        # Sonuçları dosyaya kaydet
        with open(f'{test_dir}/history_test_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        return True
    
    except Exception as e:
        print(f"Geçmiş verileri testi başarısız: {str(e)}")
        return False

def test_stats_endpoint():
    """İstatistik endpoint'ini test eder"""
    print("\nİstatistik endpoint'i test ediliyor...")
    
    try:
        response = requests.get(f'{API_URL}/stats')
        response.raise_for_status()
        result = response.json()
        
        print(f"İstatistikler alındı:")
        if 'message' in result and result['message'] == 'Henüz sensör verisi bulunmuyor':
            print("Henüz sensör verisi bulunmuyor")
        else:
            print(f"Toprak Nemi - Ortalama: {result['soil_moisture']['mean']:.2f}, Min: {result['soil_moisture']['min']:.2f}, Max: {result['soil_moisture']['max']:.2f}")
            print(f"Sıcaklık - Ortalama: {result['temperature']['mean']:.2f}, Min: {result['temperature']['min']:.2f}, Max: {result['temperature']['max']:.2f}")
            print(f"Hava Nemi - Ortalama: {result['air_humidity']['mean']:.2f}, Min: {result['air_humidity']['min']:.2f}, Max: {result['air_humidity']['max']:.2f}")
            print(f"Sulama Yüzdesi: %{result['predictions']['irrigation_percentage']:.2f}")
        
        # Sonuçları dosyaya kaydet
        with open(f'{test_dir}/stats_test_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        return True
    
    except Exception as e:
        print(f"İstatistik testi başarısız: {str(e)}")
        return False

def run_load_test():
    """API'ye yük testi yapar"""
    print("\nYük testi yapılıyor...")
    
    num_requests = 100
    request_data = {
        "soil_moisture": 500.0,
        "temperature": 28.0,
        "air_humidity": 60.0
    }
    
    start_time = time.time()
    success_count = 0
    response_times = []
    
    for i in range(num_requests):
        try:
            request_start = time.time()
            response = requests.post(
                f'{API_URL}/predict',
                json=request_data,
                headers={'Content-Type': 'application/json'}
            )
            request_end = time.time()
            
            if response.status_code == 200:
                success_count += 1
                response_times.append(request_end - request_start)
        
        except Exception:
            pass
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Sonuçları hesapla
    success_rate = (success_count / num_requests) * 100
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    requests_per_second = num_requests / total_time if total_time > 0 else 0
    
    print(f"Yük testi sonuçları:")
    print(f"Toplam istek sayısı: {num_requests}")
    print(f"Başarılı istek sayısı: {success_count}")
    print(f"Başarı oranı: %{success_rate:.2f}")
    print(f"Ortalama yanıt süresi: {avg_response_time:.4f} saniye")
    print(f"Saniyedeki istek sayısı: {requests_per_second:.2f}")
    
    # Sonuçları görselleştir
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(response_times, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Yanıt Süresi (saniye)')
    plt.ylabel('Frekans')
    plt.title('Yanıt Süresi Dağılımı')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(len(response_times)), response_times, marker='o', linestyle='-', alpha=0.7)
    plt.xlabel('İstek Numarası')
    plt.ylabel('Yanıt Süresi (saniye)')
    plt.title('İstek Başına Yanıt Süresi')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{test_dir}/load_test_results.png')
    plt.close()
    
    # Sonuçları dosyaya kaydet
    load_test_results = {
        "total_requests": num_requests,
        "successful_requests": success_count,
        "success_rate": success_rate,
        "average_response_time": avg_response_time,
        "requests_per_second": requests_per_second,
        "total_time": total_time,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f'{test_dir}/load_test_results.json', 'w') as f:
        json.dump(load_test_results, f, indent=2)
    
    return success_rate > 80  # Başarı oranı %80'in üzerindeyse test başarılı

def run_all_tests():
    """Tüm testleri çalıştırır"""
    print("Agrotopya API testleri başlatılıyor...\n")
    
    # Test sonuçlarını sakla
    test_results = {}
    
    # Sağlık kontrolü
    test_results['health'] = test_health_endpoint()
    
    # Önce simülasyon testi yap (veri oluşturmak için)
    test_results['simulate'] = test_simulate_endpoint()
    
    # Diğer testleri çalıştır
    test_results['predict'] = test_predict_endpoint()
    test_results['history'] = test_history_endpoint()
    test_results['stats'] = test_stats_endpoint()
    test_results['load_test'] = run_load_test()
    
    # Genel sonucu hesapla
    all_passed = all(test_results.values())
    
    print("\n=== Test Sonuçları ===")
    for test_name, result in test_results.items():
        status = "✅ Başarılı" if result else "❌ Başarısız"
        print(f"{test_name}: {status}")
    
    print(f"\nGenel Sonuç: {'✅ Tüm testler başarılı' if all_passed else '❌ Bazı testler başarısız'}")
    
    # Test sonuçlarını dosyaya kaydet
    with open(f'{test_dir}/test_summary.json', 'w') as f:
        json.dump({
            "test_results": test_results,
            "all_passed": all_passed,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()
