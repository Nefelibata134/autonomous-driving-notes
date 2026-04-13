# day01_bmi_complete.py

print("=====智能BMI计算器=====")
height=float(input("请输入身高（米）："))
weight=float(input("请输入体重（千克）："))

if height <=0 or weight <=0:
    print("错误，身高和体重必须大于0")
else:
    bmi=weight/(height**2)

    if bmi<18.5:
        category="偏瘦"
    elif bmi < 24:
        category = "正常"
    elif bmi < 28:
        category = "超重"
    else:
        category = "肥胖"

print(f"\n您的 BMI 值为：{bmi:.2f}")
print(f"健康等级：{category}")

if category == "偏瘦":
    advice = "建议加强营养，适当增重。"
elif category == "正常":
    advice = "保持良好的生活习惯！"
else:
    advice = "建议控制饮食，增加运动。"

print(f"健康建议：{advice}")