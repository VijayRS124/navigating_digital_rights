import 'package:flutter/material.dart';

class DashboardScreen extends StatefulWidget {
  final Map<String, dynamic> data; // Accept JSON data

  const DashboardScreen({Key? key, required this.data}) : super(key: key);

  @override
  _DashboardScreenState createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: Color.fromARGB(255, 51, 85, 255), size: 30),
          onPressed: () {
            Navigator.pop(context);
          },
        ),
      ),
      body: SingleChildScrollView(
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 20.0, vertical: 10),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Title
              Text(
                "Dashboard",
                style: TextStyle(
                  color: Color.fromARGB(255, 51, 85, 255),
                  fontWeight: FontWeight.w900,
                  fontFamily: "Roboto",
                  fontSize: 30,
                ),
              ),
              SizedBox(height: 5),
              Text(
                "Your data, your decisions",
                style: TextStyle(
                  color: Colors.black,
                  fontFamily: "Roboto",
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
              SizedBox(height: 20),

              // Displaying Input Text
              DataCard(
                title: "Input Text",
                content: widget.data["file_name"] ?? "No input provided",
              ),
              SizedBox(height: 15),

              // Displaying Prediction
              DataCard(
                title: "Prediction Result",
                content: widget.data["prediction"] ?? "No prediction available",
                highlight: true,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// Reusable Card Widget for Displaying Data
class DataCard extends StatelessWidget {
  final String title;
  final String content;
  final bool highlight;

  const DataCard({
    required this.title,
    required this.content,
    this.highlight = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: highlight ? Color.fromARGB(255, 51, 85, 255) : Colors.white,
        borderRadius: BorderRadius.circular(10),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.3),
            spreadRadius: 2,
            blurRadius: 5,
            offset: Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: TextStyle(
              color: highlight ? Colors.white : Color.fromARGB(255, 51, 85, 255),
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          SizedBox(height: 8),
          Text(
            content,
            style: TextStyle(
              fontSize: 16,
              color: highlight ? Colors.white : Colors.black87,
            ),
            softWrap: true,
          ),
        ],
      ),
    );
  }
}