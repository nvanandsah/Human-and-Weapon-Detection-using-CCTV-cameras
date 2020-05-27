package com.wellowise.lastmindeadline;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.TextView;

import java.util.ArrayList;

public class CustomAdapter extends ArrayAdapter<dataPojo> {

    public CustomAdapter(Context context, ArrayList<dataPojo> users) {
        super(context, 0, users);

    }



    @Override

    public View getView(int position, View convertView, ViewGroup parent) {

        // Get the data item for this position

        dataPojo data = getItem(position);

        // Check if an existing view is being reused, otherwise inflate the view

        if (convertView == null) {
            convertView = LayoutInflater.from(getContext()).inflate(android.R.layout.simple_list_item_1, parent, false);
        }

        // Lookup view for data population

        TextView location = (TextView) convertView.findViewById(android.R.id.text1);

        location.setText(data.Location);

        return convertView;

    }

}