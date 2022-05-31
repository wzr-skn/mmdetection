import json


def main():
    cut_number = 30000
    line = []
    with open("../../json/coco_half_person_80_train.json", "r") as f:
        for row in f.readlines():
            line.append(row)
    log = json.loads("\n".join(line))
    image_set = log["images"][:cut_number]
    annotations_set = []
    for i in range(cut_number):
        for j in range(len(log["annotations"])):
            if image_set[i]["id"] == log["annotations"][j]["image_id"]:
                annotations_set.append(log["annotations"][j])
    log["images"] = image_set
    log["annotations"] = annotations_set
    with open("../../json/coco_half_person_80_train_load.json", "w+") as fw:
        log = str(log).replace("'", '"')
        fw.write(log)

if __name__ == '__main__':
    main()