import numpy as np

read_train_file = open("Datasets/training_data.txt","r+")
read_test_file = open("Datasets/testing_data.txt","r+")

write_train_file = open("Datasets/Topic_trainingSet.txt","w") #write mode
write_test_file = open("Datasets/Topic_testSet.txt","w") #write mode

labels = []
tweets = []

unique_classes = []

count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0
count8=0
count9=0
count10=0

for line in read_train_file:
    line = line.split('\t')

    label = line[1].strip("\"#") #entity
    tweet = line[3].strip() #tweet

    #changing the entities to higher level topics.
    if(label == "هاري بوتر" or label == "بيونسيه" or label == "جستن بيبر"):
        label ="media"
        count1 = count1+1

    elif(label == "ميسي" or label == "فيدرر" or label == "ريال مدريد" or label == "برشلونة"):
        label = "sports"
        count2 = count2 + 1

    elif (label == "غوتشي" or label == "امازون"):
        label = "shopping"
        count3 = count3 + 1

    elif(label == "رمضان" or label == "الاسلام"):
        label = "religion"
        count4 = count4 + 1

    elif (label == "الارهاب" or label == "داعش"):
        label = "politics_isis"
        count5 = count5 + 1

    elif(label == "هيلاري كلنتون" or label == "اوباما" or label == "باراك أوباما" or label == "دونالد ترامب"):
        label = "politics_us"
        count6 = count6 + 1


    elif(label == "غوغل" or label == "أبل" or label == "ايفون" or label == "أندرويد" or label == "ويندوز 10" or label == "بوكيمون" or label == "جوجل"):
        label = "technology"
        count8 = count8 + 1

    elif(label == "إسرائيل" or label == "حلب" or label == "سورية" or label == "اردوغان" or label == "بشار الأسد" or
          label == "العراق" or label == "إيران" or label == "السعودية" or label == "سيسي" or label == "سوريا"):
        label = "politics_me"
        count10 = count10 + 1
    else:
        continue

    labels.append(label)
    tweets.append(tweet)

    write_train_file.write(label+"\t"+tweet+"\n")

    #if not unique_classes.__contains__(label):
    #    unique_classes.append(label)

read_train_file.close()
write_train_file.close()


for line in read_test_file:
    line = line.split('\t')

    label = line[1].strip("\"#") #entity
    tweet = line[3].strip() #tweet

    if (label == 'اليسا' or label == 'مهرجان_القاهره_السينمايي' or label == 'عادل_امام' or label == 'عمروـاديب' or label == 'فيروز'):
        label = "media"
        count1 = count1 + 1

    elif(label=='NBA' or label == 'احمد الشيخ'):
        label = "sports"
        count2 = count2 + 1

    elif (label == 'ماذا_لو_المدارس_مختلطه' or label == 'المولد النبوي الشريف'):
        label = "religion"
        count4 = count4 + 1

    elif(label=='اوبر' or label=='EgyptAir'):
        label = "Business"
        count7 = count7 + 1


    elif(label=='الصين' or label == 'كاسترو' or label=='روسيا' or label=='فرنسا' or label=='بريطانيا'):
        label = "politics_int"
        count9 = count9 + 1

    else:
        continue

    labels.append(label)
    tweets.append(tweet)

    write_test_file.write(label+"\t"+tweet+"\n")

read_test_file.close()
write_test_file.close()

print(count1, count2, count3, count4, count5, count6, count7, count8, count9, count10)
print(len(tweets))



'''

for line in read_test_file:
    line = line.split('\t')

    label = line[1].strip("\"#") #entity
    tweet = line[3] #tweet
    if not unique_classes.__contains__(label):
        unique_classes.append(label)
    #changing the entities to higher level topics.
    if(label == "هاري بوتر" or label == "بيونسيه" or label == "جستن بيبر" or label== 'اليسا' or label =='مهرجان_القاهره_السينمايي' or label == 'عادل_امام' or label =='عمروـاديب' or label == 'فيروز'
       or label=='معرض_الكويت_الدولي_للكتاب' or label=='بوب ديلان' or label =='رامي عياش' or label=='ريكي مارتن' or label =='ميريامـفارس' or label=='نادين نسيب نجيم' or label =='نوال الزغبي'):
        label ="media"
        count1 = count1+1

    elif(label=='NBA' or label == 'احمد الشيخ' or label == "ميسي" or label == "فيدرر" or label == "ريال مدريد" or label == "برشلونة"):
        label = "sports"
        count2 = count2 + 1

    elif (label == "غوتشي" or label == "امازون"):
        label = "shopping"
        count3 = count3 + 1

    elif(label=='ماذا_لو_المدارس_مختلطه' or label=='المولد النبوي الشريف' or label == "رمضان" or label == "الاسلام"):
        label = "religion"
        count4 = count4 + 1

    elif (label == "الارهاب" or label == "داعش"):
        label = "politics_isis"
        count5 = count5 + 1

    elif(label=='الولايات المتحدة' or label=='واشنطن' or label == "هيلاري كلنتون" or label == "اوباما" or label == "باراك أوباما" or label == "دونالد ترامب"):
        label = "politics_us"
        count6 = count6 + 1

    elif(label=='اوبر' or label=='EgyptAir'):
        label = "Business"
        count7 = count7 + 1

    elif(label=='SuperMarioRun' or label=='فيسبوك' or label == "غوغل" or label == "أبل" or label == "ايفون" or label == "أندرويد" or label == "ويندوز 10" or label == "بوكيمون" or label == "جوجل"):
        label = "technology"
        count8 = count8 + 1

    elif(label=='الصين' or label == 'كاسترو' or label=='روسيا' or label=='فرنسا' or label=='بريطانيا'):
        label = "politics_int"
        count9 = count9 + 1

    elif(label=='الاعلام الغربي' or label ==  'الامم_المتحده' or label=='كيري' or label == 'التحالف_العربي_سند_اليمن' or label == 'الجيش السوري' or label=='القدس' or label=='اليمن' or label =='بنغازي' or
         label == 'حماس' or label == 'ستبقى_حلب' or label == 'فلسطين' or label == "لافروف" or label=='ليبيا' or label =='الجامعه العربيه' or label =='المجتمع الدولي' or label=='الناتو' or
         label == 'النازحين السوريين' or label=='اليمين المتطرف' or label=='بوتن' or label =='بيت الوسط' or label=='جبران تويني' or label=='جبهه النصره' or label =='حزب الله' or label=='حلب الشرقيه' or
         label =='لبنان' or label=='موسكو' or label=='ميشال عون' or label=='احمد_شفيق' or label=='الاحتلال' or label == "إسرائيل" or label == "حلب" or label == "سورية" or label == "اردوغان" or
         label == "بشار الأسد" or label == "العراق" or label == "إيران" or label == "السعودية" or label == "سيسي" or label == "سوريا"):
        label = "politics_me"
        count10 = count10 + 1
    else:
        continue

    labels.append(label)
    tweets.append(tweet)


print(len(unique_classes))
print(unique_classes)

'''

