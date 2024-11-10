import pickle
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from django.http import JsonResponse


# Create your views here.

# def trangchu(requets):
#     return render(requets, 'app/test.html')

# def lab6(requets):
#     return render(requets, 'app/lab6.html')

# def lab6_thongke(requets):
#     return render(requets, 'app/lab6_thongke.html')

# views.py


def predict_price(request):
    if request.method == 'POST':
        
        location = request.POST.get('location')
        type = request.POST.get('type')
        area = float(request.POST.get('area'))
        aream2 = float(request.POST.get('aream2'))
        room_sleep = int(request.POST.get('room_sleep'))
        toilet = int(request.POST.get('toilet'))

        
        
        types = ['type_canhodichvumini','type_chungcu','type_duplex','type_nhabietthu','type_nhamatphomattien','type_nhangohem','type_nhapholienke','type_officetel','type_penthouse','type_tapthecuxa']
        type_dummies = [[1 if typ == type else 0 for typ in types]]

        locations = ['locations_phuong10_quan10','locations_phuong10_quan11','locations_phuong10_quan3','locations_phuong10_quan4','locations_phuong10_quan5','locations_phuong10_quan6','locations_phuong10_quan8','locations_phuong10_quangovap','locations_phuong10_quanphunhuan','locations_phuong10_quantanbinh','locations_phuong11_quan10','locations_phuong11_quan11','locations_phuong11_quan3','locations_phuong11_quan5','locations_phuong11_quan6','locations_phuong11_quan8','locations_phuong11_quanbinhthanh','locations_phuong11_quangovap','locations_phuong11_quanphunhuan','locations_phuong11_quantanbinh','locations_phuong12_quan10','locations_phuong12_quan11','locations_phuong12_quan3','locations_phuong12_quan5','locations_phuong12_quan6','locations_phuong12_quan8','locations_phuong12_quanbinhthanh','locations_phuong12_quangovap','locations_phuong12_quantanbinh','locations_phuong13_quan10','locations_phuong13_quan11','locations_phuong13_quan3','locations_phuong13_quan4','locations_phuong13_quan5','locations_phuong13_quan6','locations_phuong13_quan8','locations_phuong13_quanbinhthanh','locations_phuong13_quangovap','locations_phuong13_quanphunhuan','locations_phuong13_quantanbinh','locations_phuong14_quan10','locations_phuong14_quan11','locations_phuong14_quan3','locations_phuong14_quan4','locations_phuong14_quan5','locations_phuong14_quan6','locations_phuong14_quan8','locations_phuong14_quanbinhthanh','locations_phuong14_quangovap','locations_phuong14_quantanbinh','locations_phuong15_quan10','locations_phuong15_quan11','locations_phuong15_quan4','locations_phuong15_quan8','locations_phuong15_quanbinhthanh','locations_phuong15_quangovap','locations_phuong15_quanphunhuan','locations_phuong15_quantanbinh','locations_phuong16_quan11','locations_phuong16_quan4','locations_phuong16_quan8','locations_phuong16_quangovap','locations_phuong17_quanbinhthanh','locations_phuong17_quangovap','locations_phuong17_quanphunhuan','locations_phuong18_quan4','locations_phuong19_quanbinhthanh','locations_phuong1_quan10','locations_phuong1_quan11','locations_phuong1_quan3','locations_phuong1_quan4','locations_phuong1_quan5','locations_phuong1_quan6','locations_phuong1_quan8','locations_phuong1_quanbinhthanh','locations_phuong1_quangovap','locations_phuong1_quanphunhuan','locations_phuong1_quantanbinh','locations_phuong21_quanbinhthanh','locations_phuong22_quanbinhthanh','locations_phuong24_quanbinhthanh','locations_phuong25_quanbinhthanh','locations_phuong26_quanbinhthanh','locations_phuong27_quanbinhthanh','locations_phuong28_quanbinhthanh','locations_phuong2_quan10','locations_phuong2_quan11','locations_phuong2_quan3','locations_phuong2_quan4','locations_phuong2_quan5','locations_phuong2_quan6','locations_phuong2_quan8','locations_phuong2_quanbinhthanh','locations_phuong2_quanphunhuan','locations_phuong2_quantanbinh','locations_phuong3_quan11','locations_phuong3_quan3','locations_phuong3_quan4','locations_phuong3_quan5','locations_phuong3_quan6','locations_phuong3_quan8','locations_phuong3_quanbinhthanh','locations_phuong3_quangovap','locations_phuong3_quanphunhuan','locations_phuong3_quantanbinh','locations_phuong4_quan10','locations_phuong4_quan11','locations_phuong4_quan3','locations_phuong4_quan4','locations_phuong4_quan5','locations_phuong4_quan6','locations_phuong4_quan8','locations_phuong4_quangovap','locations_phuong4_quanphunhuan','locations_phuong4_quantanbinh','locations_phuong5_quan10','locations_phuong5_quan11','locations_phuong5_quan3','locations_phuong5_quan5','locations_phuong5_quan6','locations_phuong5_quan8','locations_phuong5_quanbinhthanh','locations_phuong5_quangovap','locations_phuong5_quanphunhuan','locations_phuong5_quantanbinh','locations_phuong6_quan10','locations_phuong6_quan11','locations_phuong6_quan4','locations_phuong6_quan5','locations_phuong6_quan6','locations_phuong6_quan8','locations_phuong6_quanbinhthanh','locations_phuong6_quangovap','locations_phuong6_quantanbinh','locations_phuong7_quan10','locations_phuong7_quan11','locations_phuong7_quan5','locations_phuong7_quan6','locations_phuong7_quan8','locations_phuong7_quanbinhthanh','locations_phuong7_quangovap','locations_phuong7_quanphunhuan','locations_phuong7_quantanbinh','locations_phuong8_quan10','locations_phuong8_quan11','locations_phuong8_quan4','locations_phuong8_quan5','locations_phuong8_quan6','locations_phuong8_quan8','locations_phuong8_quangovap','locations_phuong8_quanphunhuan','locations_phuong8_quantanbinh','locations_phuong9_quan10','locations_phuong9_quan11','locations_phuong9_quan3','locations_phuong9_quan4','locations_phuong9_quan5','locations_phuong9_quan6','locations_phuong9_quan8','locations_phuong9_quangovap','locations_phuong9_quanphunhuan','locations_phuong9_quantanbinh','locations_phuongankhanhquan2cu_thanhphothuduc','locations_phuonganlac_quanbinhtan','locations_phuonganlaca_quanbinhtan','locations_phuonganloidongquan2cu_thanhphothuduc','locations_phuonganphudong_quan12','locations_phuonganphuquan2cu_thanhphothuduc','locations_phuongbennghe_quan1','locations_phuongbenthanh_quan1','locations_phuongbinhchieuquanthuduccu_thanhphothuduc','locations_phuongbinhhunghoa_quanbinhtan','locations_phuongbinhhunghoaa_quanbinhtan','locations_phuongbinhhunghoab_quanbinhtan','locations_phuongbinhthoquanthuduccu_thanhphothuduc','locations_phuongbinhthuan_quan7','locations_phuongbinhtridong_quanbinhtan','locations_phuongbinhtridonga_quanbinhtan','locations_phuongbinhtridongb_quanbinhtan','locations_phuongbinhtrungdongquan2cu_thanhphothuduc','locations_phuongbinhtrungtayquan2cu_thanhphothuduc','locations_phuongcatlaiquan2cu_thanhphothuduc','locations_phuongcaukho_quan1','locations_phuongcauonglanh_quan1','locations_phuongcogiang_quan1','locations_phuongdakao_quan1','locations_phuongdonghungthuan_quan12','locations_phuonghiepbinhchanhquanthuduccu_thanhphothuduc','locations_phuonghiepbinhphuocquanthuduccu_thanhphothuduc','locations_phuonghiepphuquan9cu_thanhphothuduc','locations_phuonghieptan_quantanphu','locations_phuonghiepthanh_quan12','locations_phuonghoathanh_quantanphu','locations_phuonglinhchieuquanthuduccu_thanhphothuduc','locations_phuonglinhdongquanthuduccu_thanhphothuduc','locations_phuonglinhtayquanthuduccu_thanhphothuduc','locations_phuonglinhtrungquanthuduccu_thanhphothuduc','locations_phuonglinhxuanquanthuduccu_thanhphothuduc','locations_phuonglongbinhquan9cu_thanhphothuduc','locations_phuonglongphuocquan9cu_thanhphothuduc','locations_phuonglongthanhmyquan9cu_thanhphothuduc','locations_phuonglongtruongquan9cu_thanhphothuduc','locations_phuongnguyencutrinh_quan1','locations_phuongnguyenthaibinh_quan1','locations_phuongphamngulao_quan1','locations_phuongphuhuuquan9cu_thanhphothuduc','locations_phuongphumy_quan7','locations_phuongphuocbinhquan9cu_thanhphothuduc','locations_phuongphuoclongaquan9cu_thanhphothuduc','locations_phuongphuoclongbquan9cu_thanhphothuduc','locations_phuongphuthanh_quantanphu','locations_phuongphuthohoa_quantanphu','locations_phuongphuthuan_quan7','locations_phuongphutrung_quantanphu','locations_phuongsonky_quantanphu','locations_phuongtambinhquanthuduccu_thanhphothuduc','locations_phuongtamphuquanthuduccu_thanhphothuduc','locations_phuongtanchanhhiep_quan12','locations_phuongtandinh_quan1','locations_phuongtangnhonphuaquan9cu_thanhphothuduc','locations_phuongtangnhonphubquan9cu_thanhphothuduc','locations_phuongtanhung_quan7','locations_phuongtanhungthuan_quan12','locations_phuongtankieng_quan7','locations_phuongtanphong_quan7','locations_phuongtanphu_quan7','locations_phuongtanphuquan9cu_thanhphothuduc','locations_phuongtanquy_quan7','locations_phuongtanquy_quantanphu','locations_phuongtansonnhi_quantanphu','locations_phuongtantao_quanbinhtan','locations_phuongtantaoa_quanbinhtan','locations_phuongtanthanh_quantanphu','locations_phuongtanthoihiep_quan12','locations_phuongtanthoihoa_quantanphu','locations_phuongtanthoinhat_quan12','locations_phuongtanthuandong_quan7','locations_phuongtanthuantay_quan7','locations_phuongtaythanh_quantanphu','locations_phuongthanhloc_quan12','locations_phuongthanhmyloiquan2cu_thanhphothuduc','locations_phuongthanhxuan_quan12','locations_phuongthaodienquan2cu_thanhphothuduc','locations_phuongthoian_quan12','locations_phuongthuthiemquan2cu_thanhphothuduc','locations_phuongtrungmytay_quan12','locations_phuongtruongthanhquan9cu_thanhphothuduc','locations_phuongtruongthoquanthuduccu_thanhphothuduc','locations_phuongvothisau_quan3','locations_thitrancanthanh_huyencangio','locations_thitrancuchi_huyencuchi','locations_thitranhocmon_huyenhocmon','locations_thitrannhabe_huyennhabe','locations_thitrantantuc_huyenbinhchanh','locations_xaannhontay_huyencuchi','locations_xaanphu_huyencuchi','locations_xaanphutay_huyenbinhchanh','locations_xaanthoidong_huyencangio','locations_xabadiem_huyenhocmon','locations_xabinhchanh_huyenbinhchanh','locations_xabinhhung_huyenbinhchanh','locations_xabinhkhanh_huyencangio','locations_xabinhloi_huyenbinhchanh','locations_xabinhmy_huyencuchi','locations_xadaphuoc_huyenbinhchanh','locations_xadongthanh_huyenhocmon','locations_xahiepphuoc_huyennhabe','locations_xahoaphu_huyencuchi','locations_xahunglong_huyenbinhchanh','locations_xaleminhxuan_huyenbinhchanh','locations_xalonghoa_huyencangio','locations_xalongthoi_huyennhabe','locations_xanhibinh_huyenhocmon','locations_xanhonduc_huyennhabe','locations_xanhuanduc_huyencuchi','locations_xaphamvancoi_huyencuchi','locations_xaphamvanhai_huyenbinhchanh','locations_xaphongphu_huyenbinhchanh','locations_xaphuhoadong_huyencuchi','locations_xaphumyhung_huyencuchi','locations_xaphuochiep_huyencuchi','locations_xaphuockien_huyennhabe','locations_xaphuocloc_huyennhabe','locations_xaphuocthanh_huyencuchi','locations_xaphuocvinhan_huyencuchi','locations_xaphuxuan_huyennhabe','locations_xaquyduc_huyenbinhchanh','locations_xatananhoi_huyencuchi','locations_xatanhiep_huyenhocmon','locations_xatankien_huyenbinhchanh','locations_xatannhut_huyenbinhchanh','locations_xatanphutrung_huyencuchi','locations_xatanquytay_huyenbinhchanh','locations_xatanthanhdong_huyencuchi','locations_xatanthanhtay_huyencuchi','locations_xatanthoinhi_huyenhocmon','locations_xatanthonghoi_huyencuchi','locations_xatanxuan_huyenhocmon','locations_xathaimy_huyencuchi','locations_xathoitamthon_huyenhocmon','locations_xatrungan_huyencuchi','locations_xatrungchanh_huyenhocmon','locations_xatrunglapha_huyencuchi','locations_xatrunglapthuong_huyencuchi','locations_xavinhloca_huyenbinhchanh','locations_xavinhlocb_huyenbinhchanh','locations_xaxuanthoidong_huyenhocmon','locations_xaxuanthoison_huyenhocmon','locations_xaxuanthoithuong_huyenhocmon']
        location_dummies = [[1 if loc == location else 0 for loc in locations]]

        # chia data
        scaler_data_new = [[area, aream2, room_sleep, toilet]]
        # types_dummies_data_new = [type_dummies]
        # locations_dummies_data_new = [location_dummies]
        # types_dummies_data_new = np.array(type_dummies).reshape(1, -1)  # Reshape thành (1, 10)
        # locations_dummies_data_new = np.array(location_dummies).reshape(1, -1)
        # all_features = [area, aream2, room_sleep, toilet] + type_dummies + location_dummies



        
        # load model and scaler
        model_tree = pickle.load(open('G:\My Drive\IT LOR\Phương Pháp Phát Triển Phần Mềm Hướng Đối Tượng\locations_model_dinhgia_data_tong_scaler_tree','rb'))
        scaler = pickle.load(open('G:\My Drive\IT LOR\Phương Pháp Phát Triển Phần Mềm Hướng Đối Tượng\scaler_model', 'rb'))
            
        # scaler data new
        data_new_scaler = scaler.transform(scaler_data_new)
        # data_new = np.concatenate((data_new_scaler,types_dummies_data_new),axis=1)
        # data_new2 = np.concatenate((data_new,locations_dummies_data_new),axis=1)
        data_new = np.concatenate((data_new_scaler, type_dummies, location_dummies), axis=1)
        new_predict = model_tree.predict(data_new)
        predict_formatted = '{:,.0f}'.format(new_predict[0]).replace(',', '.')

        return JsonResponse({
            'predict': predict_formatted,
        })
    else:
        return render(request, 'app/index.html')
    
    
    
def visualize_data(request):
    df = pd.read_csv('G:\My Drive\IT LOR\Phương Pháp Phát Triển Phần Mềm Hướng Đối Tượng\data_moi_nhat.csv', index_col=0)
    # load model and scaler
    model_tree = pickle.load(open('G:\My Drive\IT LOR\Phương Pháp Phát Triển Phần Mềm Hướng Đối Tượng\locations_model_dinhgia_data_tong_scaler_tree','rb'))
    scaler = pickle.load(open('G:\My Drive\IT LOR\Phương Pháp Phát Triển Phần Mềm Hướng Đối Tượng\scaler_model', 'rb'))
    
    chart = []
    
    # # aream2 vs price
    # plt.figure()
    # sns.scatterplot(x='price', y='area/m2', data=df)
    # plt.title('Biểu đồ tương quan giữa diện tích/m2 và giá')
    # chart.append(get_chart_image(plt))

    # #area vs price
    # plt.figure()
    # sns.scatterplot(x='area', y='price', data=df)
    # plt.title('Biểu đồ tương quan giữa diện tích và giá')
    # chart.append(get_chart_image(plt))


    #  so sánh
    X= df[['area','area/m2','room_sleep','toilet','type','locations']]
    y= df['price']
    scaler_col = ['area','area/m2','room_sleep','toilet']
    dummies_col = ['type','locations']
    X_dummies = pd.get_dummies(X[dummies_col]).astype(int)
    X[scaler_col] = scaler.fit_transform(X[scaler_col])
    X_final = pd.concat([X[scaler_col],X_dummies],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_final, y,test_size=0.2,random_state=54)
    y_pred = model_tree.predict(X_test)
    y_test_series = pd.Series(y_test, name='price')
    y_pred_series = pd.Series(y_pred,index=y_test_series.index, name='price')
    y_test_series = scaler.fit_transform(y_test_series.to_numpy().reshape((-1, 1)))
    y_pred_series = scaler.fit_transform(y_pred_series.to_numpy().reshape((-1, 1)))
    plt.figure(figsize=(18,10))
    plt.plot(y_test_series[:200], label='Giá trị thực tế', color='b', linewidth=1)
    plt.plot(y_pred_series[:200], label='Giá trị dự đoán', color='g', linewidth=1) 
    plt.title('So sánh giá trị dự đoán và thực tế mô hình định giá ')
    plt.xlabel('Mẫu dữ liệu')
    plt.ylabel('Giá trị')
    plt.legend()
    chart.append(get_chart_image(plt))


    #toilet vs price
    plt.figure(figsize=(18,10))
    sns.barplot(x='toilet', y='price', data=df)
    plt.title('Biểu đồ tương quan giữa số lượng toilet và giá')
    chart.append(get_chart_image(plt))
    
    # room_sleep vs price
    plt.figure(figsize=(18,10))
    sns.barplot(x = df['room_sleep'], y = df['price'])
    plt.xlabel('Phòng ngủ')
    plt.ylabel('Giá')
    plt.title('Biểu đồ tương quan giữa số lượng phòng ngủ và giá')
    chart.append(get_chart_image(plt))

    # # location vs price
    # plt.figure()
    # sns.barplot(x = df['location'], y = df['price'])
    # plt.xlabel('Quận/Huyện')
    # plt.ylabel('Giá')
    # plt.title('Biểu đồ tương quan giữa Quận/Huyện và giá')
    # chart.append(get_chart_image(plt))
    
    # ma trận tương quan
    matrix = df[['area','area/m2','room_sleep','toilet','type_encoder','locations_encoder','price']].corr()
    plt.figure(figsize=(18,10))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap="coolwarm", square=True)
    plt.title('Ma trận tương quan dữ liệu đầu vào')
    chart.append(get_chart_image(plt))
    
    

    

    context = {'charts': chart}
    return render(request, 'app/visualize.html', context)

def get_chart_image(plt):
    """Hàm hỗ trợ để chuyển biểu đồ matplotlib thành base64"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_base64