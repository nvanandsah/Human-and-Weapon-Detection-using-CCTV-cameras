package com.wellowise.lastmindeadline;

public class dataPojo {
    public String ImageURL;
    public String Location;
    public String TimeStamp;
    public String key;
    public dataPojo(String imageURL, String Location, String timestamp, String key) {
        this.ImageURL = imageURL;
        this.TimeStamp = timestamp;
        this.Location = Location;
        this.key = key;
    }

}
