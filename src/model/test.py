from concrete import Model

model = Model()
result = model.analyze(['今日は晴れの日、洗濯日和です', 'なにもかもが失敗続きだ'])
print(type(result))
print(result)