import 'package:flutter/material.dart';

void main() {
  runApp(GreenScoreApp());
}

class GreenScoreApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Green Score',
      theme: ThemeData(primarySwatch: Colors.green),
      home: HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  String? selectedTransport;
  int greenScore = 0;

  void calculateScore() {
    if (selectedTransport == "Bus") {
      greenScore = 10;
    } else if (selectedTransport == "Metro") {
      greenScore = 15;
    } else if (selectedTransport == "Cycle") {
      greenScore = 25;
    } else if (selectedTransport == "EV") {
      greenScore = 20;
    }
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Green Score Tracker")),
      body: Padding(
        padding: EdgeInsets.all(20),
        child: Column(
          children: [
            DropdownButton<String>(
              hint: Text("Select Mode of Transport"),
              value: selectedTransport,
              onChanged: (String? newValue) {
                setState(() {
                  selectedTransport = newValue;
                });
              },
              items: ["Bus", "Metro", "Cycle", "EV"]
                  .map((mode) => DropdownMenuItem(
                        value: mode,
                        child: Text(mode),
                      ))
                  .toList(),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: calculateScore,
              child: Text("Calculate Green Score"),
            ),
            SizedBox(height: 20),
            Text("Your Green Score: $greenScore"),
          ],
        ),
      ),
    );
  }
}
