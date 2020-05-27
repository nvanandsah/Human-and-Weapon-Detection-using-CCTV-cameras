package com.wellowise.lastmindeadline;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ListView;

import com.google.firebase.database.ChildEventListener;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

import java.sql.Time;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    FirebaseDatabase database;
    DatabaseReference db;
    ArrayList<dataPojo> array;
    CustomAdapter adapter;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ListView listView = findViewById(R.id.list_view);
        database = FirebaseDatabase.getInstance();
        db = database.getReference();
        array = new ArrayList<>();
        adapter = new CustomAdapter(this, array);
        listView.setAdapter(adapter);
        listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> adapterView, View view, int i, long l) {
                dataPojo data = array.get(i);
                Log.d("Position", "" + i  +" " + l);
                Intent newIntent = new Intent(MainActivity.this, ImageActivity.class);
                Log.d("URL", data.ImageURL);
                newIntent.putExtra("URL", data.ImageURL);
                startActivity(newIntent);
            }
        });
        retrieve();
    }

    public void retrieve() {
        db.addChildEventListener(new ChildEventListener() {
            @Override
            public void onChildAdded(DataSnapshot dataSnapshot, String s) {
                fetchData(dataSnapshot);
            }

            @Override
            public void onChildChanged(DataSnapshot dataSnapshot, String s) {
                fetchData(dataSnapshot);
            }

            @Override
            public void onChildRemoved(DataSnapshot dataSnapshot) {

            }

            @Override
            public void onChildMoved(DataSnapshot dataSnapshot, String s) {

            }

            @Override
            public void onCancelled(DatabaseError databaseError) {

            }
        });
    }

    private void fetchData(DataSnapshot dataSnapshot)
    {
        dataPojo data = new dataPojo((String)dataSnapshot.child("ImageURL").getValue(), (String)dataSnapshot.child("Location").getValue(), dataSnapshot.child("TimeStamp").getValue().toString(), dataSnapshot.getKey());
        Log.d("Data Snapshot", dataSnapshot.getKey());
        for (int i = 0; i <array.size(); i++) {
            if (array.get(i).key.equals(dataSnapshot.getKey())) {
                array.set(i, data);
                adapter.notifyDataSetChanged();
                Log.d("URL", data.Location);
                return ;
            }
        }
        array.add(data);
        Log.d("URL", data.Location);
        /*for (DataSnapshot ds : dataSnapshot.getChildren())
        {
            Log.d("Data", ds.getValue().toString());
            String location = ds.getValue(dataPojo.class).Location;
            String ImageURL = ds.getValue(dataPojo.class).ImageURL;
            Long TimeStamp = ds.getValue(dataPojo.class).TimeStamp;
            array.add(new dataPojo(ImageURL, location, TimeStamp));
        }*/
        adapter.notifyDataSetChanged();

    }

}
