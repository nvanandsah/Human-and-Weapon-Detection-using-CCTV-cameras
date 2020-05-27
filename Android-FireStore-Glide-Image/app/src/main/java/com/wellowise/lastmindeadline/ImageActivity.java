package com.wellowise.lastmindeadline;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;

public class ImageActivity extends AppCompatActivity {
    ImageView img;
    String url;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image);
        img = findViewById(R.id.imageView);
        Intent intent = getIntent();
        url = intent.getStringExtra("URL");
        Log.d("URL", url);
        FirebaseStorage storage = FirebaseStorage.getInstance();
        StorageReference ref = storage.getReferenceFromUrl(url);
        GlideApp.with(getApplicationContext())
                .load(ref)
                .into(img);
    }

}
