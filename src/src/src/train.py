keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from src.dataset import prepare_data, get_augmentor
from src.model import build_model
from src.utils import get_class_weights




def train(csv_path='data/fer2013.csv', model_dir='saved_models', batch_size=64, epochs=50):
os.makedirs(model_dir, exist_ok=True)
X_train, X_val, X_test, y_train, y_val, y_test, y_train_raw, y_val_raw, y_test_raw = prepare_data(csv_path, image_size=(48,48))
datagen = get_augmentor()
datagen.fit(X_train)


model = build_model(input_shape=(48,48,1), num_classes=y_train.shape[1], weights='imagenet')


ckpt_path = os.path.join(model_dir, 'fer_mobilenet_best.h5')
ckpt = ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1)
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)


class_weights = get_class_weights(np.argmax(y_train, axis=1))


steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
history = model.fit(
datagen.flow(X_train, y_train, batch_size=batch_size),
validation_data=(X_val, y_val),
epochs=epochs,
steps_per_epoch=steps_per_epoch,
callbacks=[ckpt, rlr, es],
class_weight=class_weights
)


# fine-tune: unfreeze some layers
base = model.layers[3]
base.trainable = True
for layer in base.layers[:-20]:
layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
history_ft = model.fit(
datagen.flow(X_train, y_train, batch_size=batch_size),
validation_data=(X_val, y_val),
epochs=20,
steps_per_epoch=steps_per_epoch,
callbacks=[ckpt, rlr, es],
class_weight=class_weights
)


final_path = os.path.join(model_dir, 'fer_mobilenet_final.h5')
model.save(final_path)
print('Saved final model to', final_path)
return model




if __name__ == '__main__':
train(csv_path=args.csv, model_dir=args.model_dir, batch_size=args.batch, epochs=args.epochs)
