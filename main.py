import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


if 'img_bytes' not in st.session_state:
    st.session_state['img_bytes'] = None

WEIGHT_DIR = os.getcwd() + "/Googlenetmodel.pth" 

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.googlenet(pretrained=True) 
model.fc = nn.Linear(in_features=1024, out_features=4)  
model.load_state_dict(torch.load(WEIGHT_DIR, map_location=torch.device(device)))
model = model.to(device)

def inference(picture):

  a = "枯萎病"
  b = "普通锈斑病"
  c = "灰叶斑病"
  d = "健康"

  img = Image.open(picture).convert('RGB') 

  transformations = transforms.Compose([transforms.Resize(size = (256,256)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                            mean = [0.485, 0.456, 0.406],
                            std  = [0.229, 0.224, 0.225],
                    ),                      
  ])

  img_tensor = transformations(img).reshape(1,3,256,256).to(device)

  model.eval()

  with torch.no_grad():
      
      output = model(img_tensor)
      output = output.argmax(1).item()

  if output == 0:
     return a
  elif output == 1:
     return b
  elif output == 2:
     return c
  else:
     return d


st.title("玉米叶片病害识别系统")

tab1, tab2, tab3 = st.tabs(["使用拍摄进行识别", "输入图片进行识别", "解决方案"])

with tab1:
  st.subheader('使用拍摄进行识别')

  picture = st.camera_input("")
  if picture:
    st.write(inference(picture = picture))

with tab2:
  st.subheader('输入图片进行识别') 
  uploaded_file = st.file_uploader("请选择一张图片")

  if uploaded_file is not None:
    st.session_state['img_bytes'] = uploaded_file.getvalue()

  if st.session_state['img_bytes'] != None:
    st.image(st.session_state['img_bytes'])

  picture = uploaded_file
  if picture:
    st.write(inference(picture = picture))

with tab3:
  st.subheader('解决方案')

  if(st.button("枯萎病")):st.write("""1、减少菌源: 发病田秸秆不要进行粉碎还田, 可将病株集中烧毁并对土壤进行深翻减少田间菌源量。
                                \n2、合理密植: 请按照产品包装袋推荐密度播种。
                                \n3、加强田间管理: 结合土壤深松和宽窄行播种技术, 增强根系活力和田间透风透光水平; 注意平衡施肥, 避免偏施氮肥, 适当增施钾肥和锌肥可增强玉米抗病能力; 雨后要及时排水。
                                \n4、治虫防病: 及时防治地下害虫粘虫及玉米螟等可以造成伤口的虫害, 减少病原菌侵染玉米的机会。提早防治地下害虫和地上玉米螟咬食, 减少因伤口浸染病原菌而发病造成干枯的可能。
                                \n5、合理用药: 在玉米抽穗期发病即喷洒药剂防治, 每隔7-10天防治1次, 连续防治1-2次。药剂可选用50%多菌灵可湿性粉剂1000倍液, 或50%甲基硫菌灵可湿性粉剂1000倍液。可以用多菌灵及甲基硫菌灵防治, 玉米干枯病主要是做好灌溉, 肥料施足不缺肥, 低洼地及时排除田间积水。
                                """)
  if(st.button("普通锈斑病")):st.write("""1、选择25%三唑酮可湿性粉剂1500-2000倍液、或25%粉锈宁可湿性粉剂1500倍液、或40%氟硅唑乳油8000倍液进行喷防, 每隔7天左右喷施1次, 连喷2-3次, 用药最好轮换使用。
                                      \n2、田间已经大面积发生, 可选择25%三唑酮可湿性粉剂、或25%粉锈宁可湿性粉剂、或25%敌立脱乳油、12.5%烯唑醇等药剂, 按照要求稀释后, 进行喷雾防治, 根据发病情况确定喷雾次数。
                                      \n3、推荐氟环唑, 12.5%的氟环唑悬浮剂30ml一亩地, 能快速锁住病斑, 避免蔓延; 还可以使用30%的苯甲.丙环唑乳油防治, 推荐使用20到25毫升一亩地, 效果也是非常棒的。
""")
  if(st.button("灰叶斑病")):st.write("""1、选用抗病或耐病的品种。选用适合当地种植、丰产性好、抗玉米灰斑病的优良品种, 是保证玉米高产稳产的基本条件。
                                      \n2、加强田间管理。在容易发生灰斑病的地块, 适当降低密度, 或者采用不同作物间作, 比如大豆和玉米间作的方式, 可以有效的增加通风效率, 降低田间空气湿度, 改善田间小气候。而且, 还要加强田间管理, 对于农田的供排水设施进行及时的维护, 雨后及时排水, 防止湿气滞留, 这也是可以在一定程度上降低灰斑病的发生。
                                      \n3、进行大面积轮作。尤其是发病地块, 需要合理布局作物与品种, 定期轮换, 减少玉米灰斑病病菌侵染源。再结合合理的施肥, 一般每亩施普钙50公斤, 硫酸钾10公斤。一方面可以改善农田土壤理化性状, 降低土壤病原菌数量, 提高植物长势。另外, 轮作还能改变土壤的微生态环境, 调控微生物的分布, 也是促进植物生长, 降低灰斑病发生的重要途径。
                                      \n4、清除病原菌。玉米收获后, 要及时清除遗留在田间地块中的玉米秸秆、病叶等残体, 尤其是堆过秸秆的重病地块, 应彻底清除, 并在雨季开始前处理完毕; 使用玉米秆叶堆沤农家肥的, 肥料要充分腐熟后才能施用于地块、田间, 以便消灭和减少菌源, 这也可以有效地从源头上抑制灰斑病的发生。
                                      \n5、药剂防治。应遵循“预防为主,  综合防治”的方针。根据发病特点, 主要在玉米大喇叭口期、抽雄抽穗期和灌浆初期3个关键时期进行药物防治。选择对玉米灰斑病防治效果较好的药剂, 主要有25%丙环唑可湿性粉剂135毫升/公顷兑水喷雾; 10%苯醚甲环唑450克/公顷兑水喷雾; 75%三环唑（或稻瘟净）+农用链霉素+70%甲基托布津(或多菌灵)各1/2袋混合兑水喷雾。5-7天防治一次, 连续用药2-3次, 施药时要注意喷匀喷透, 若喷后1-2小时内遇雨应重喷, 确保防治效果。
                                      """)