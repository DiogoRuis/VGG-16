import tensorflow as tf
from tf_keras.applications import VGG16
from tf_keras.layers import Dense, GlobalAveragePooling2D
from tf_keras.models import Model
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.optimizers import Adam


altura_img, largura_img = 224, 224
tamanho_batch = 15 
num_classes = 4  


diretorio_treino = 'data'
diretorio_validacao = 'validacao'
diretorio_teste = 'test'


modelo_base = VGG16(weights='imagenet', include_top=False, input_shape=(altura_img, largura_img, 3))


for camada in modelo_base.layers:
    camada.trainable = False


x = modelo_base.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
predicoes = Dense(num_classes, activation='softmax')(x)


modelo = Model(inputs=modelo_base.input, outputs=predicoes)


modelo.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


gerador_treino = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

gerador_teste = ImageDataGenerator(rescale=1.0/255)

gerador_treino_fluxo = gerador_treino.flow_from_directory(
    diretorio_treino,
    target_size=(altura_img, largura_img),
    batch_size=tamanho_batch,
    class_mode='categorical'
)

gerador_validacao_fluxo = gerador_teste.flow_from_directory(
    diretorio_validacao,
    target_size=(altura_img, largura_img),
    batch_size=tamanho_batch,
    class_mode='categorical'
)


modelo.fit(
    gerador_treino_fluxo,
    steps_per_epoch=gerador_treino_fluxo.samples // tamanho_batch,
    validation_data=gerador_validacao_fluxo,
    validation_steps=gerador_validacao_fluxo.samples // tamanho_batch,
    epochs=10
)


gerador_teste_fluxo = gerador_teste.flow_from_directory(
    diretorio_teste,
    target_size=(altura_img, largura_img),
    batch_size=tamanho_batch,
    class_mode='categorical'
)

perda, acuracia = modelo.evaluate(gerador_teste_fluxo)
print(f"Acurácia no Teste: {acuracia * 100:.2f}%")