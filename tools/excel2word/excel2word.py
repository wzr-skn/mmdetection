# # coding：<encoding name> ： # coding: utf-8
# import xlrd
# from docxtpl import DocxTemplate
#
#
# def main():
#     data = xlrd.open_workbook(r'/media/traindata/excel2word/test.xlsx')
#     for x in range(0, 1):
#         print(x)
#         table = data.sheets()[x]
#         names = data.sheet_names()[x]  # 获取表名
#         nrows = table.nrows
#         print(names)
#
#         for i in range(nrows):
#             if i == 0:  # 跳过表头信息
#                 continue
#             poverty_people = table.row_values(i)[0]  # 贫困户名字
#             helper = table.row_values(i)[1]  # 帮扶人名字
#             helper_tel = table.row_values(i)[2]  # 帮扶人联系方式
#             helper_tel = str(helper_tel).replace(".0", "")  # 去掉数字中的.0
#             context = {
#                 "helper": helper,
#                 "helper_tel": helper_tel,
#             }
#             word = DocxTemplate(r'/media/traindata/excel2word/贫困户扶持表.docx')
#             print(context)
#             word.render(context)
#             word.save("{}".format(poverty_people) + ".docx")
#         print("保存成功")
#
#
# if __name__ == '__main__':
#     main()


# coding：<encoding name> ： # coding: utf-8
import xlrd
from docxtpl import DocxTemplate


def main():
    data = xlrd.open_workbook(r'/media/traindata/excel2word/2022年包保人一览表 - 副本.xlsx')
    for x in range(0, 1):
        print(x)
        table = data.sheets()[x]
        names = data.sheet_names()[x]  # 获取表名
        nrows = table.nrows
        print(names)

        for i in range(nrows):
            if i == 0:  # 跳过表头信息
                continue
            poverty_people = table.row_values(i)[4].split("、")  # 贫困户名字
            print(len(poverty_people))
            helper = table.row_values(i)[1]  # 帮扶人名字
            helper_tel = table.row_values(i)[3]  # 帮扶人联系方式
            helper_tel = str(helper_tel).replace(".0", "")  # 去掉数字中的.0
            context = {
                "helper": helper,
                "helper_tel": helper_tel,
            }
            for j in range(len(poverty_people)):

                word = DocxTemplate(r'/media/traindata/excel2word/防返贫监测帮扶工作“明白卡”.docx')
                print(context)
                word.render(context)
                if len(poverty_people[j]) > 5:
                    word.save("{}".format(poverty_people[j][:-6]) + ".docx")
                else:
                    word.save("{}".format(poverty_people[j]) + ".docx")
        print("保存成功")


if __name__ == '__main__':
    main()