const int EnableL = 5;
const int HighL = 6;       // LEFT SIDE MOTOR
const int LowL =7;

const int EnableR = 10;
const int HighR = 8;       //RIGHT SIDE MOTOR
const int LowR =9;

const int D0 = 0;       //Raspberry pin 21    LSB
const int D1 = 1;       //Raspberry pin 22
const int D2 = 2;       //Raspberry pin 23
const int D3 = 3;       //Raspberry pin 24    MSB

int a,b,c,d,data;

void setup() 
{
  pinMode(EnableL, OUTPUT);      
  pinMode(HighL, OUTPUT);
  pinMode(LowL, OUTPUT);
  
  pinMode(EnableR, OUTPUT);
  pinMode(HighR, OUTPUT);
  pinMode(LowR, OUTPUT);
  
  pinMode(D0, INPUT_PULLUP);
  pinMode(D1, INPUT_PULLUP);
  pinMode(D2, INPUT_PULLUP);
  pinMode(D3, INPUT_PULLUP);
}

void Data()
{
   a = digitalRead(D0);
   b = digitalRead(D1);
   c = digitalRead(D2);
   d = digitalRead(D3);

   data = 8*d+4*c+2*b+a;
}

void Stop()
{
  digitalWrite(HighL, LOW);
  digitalWrite(LowL, HIGH);
  analogWrite(EnableL,0);

  digitalWrite(HighR, LOW);
  digitalWrite(LowR, HIGH);
  analogWrite(EnableR,0);
}

void Forward()
{
  digitalWrite(HighL, HIGH);
  digitalWrite(LowL, LOW);
  analogWrite(EnableL,80);

  digitalWrite(HighR, HIGH);
  digitalWrite(LowR, LOW);
  analogWrite(EnableR,80); 
}

void Backward()
{
  digitalWrite(HighL, LOW);
  digitalWrite(LowL, HIGH);
  analogWrite(EnableL,80);

  digitalWrite(HighR, LOW);
  digitalWrite(LowR, HIGH);
  analogWrite(EnableR,80);  
}

void Left1()
{
  digitalWrite(HighL, HIGH);
  digitalWrite(LowL, LOW);
  analogWrite(EnableL,50);

  digitalWrite(HighR, HIGH);
  digitalWrite(LowR, LOW);
  analogWrite(EnableR,90); 
} 

void Left2()
{
  digitalWrite(HighL, HIGH);
  digitalWrite(LowL, LOW);
  analogWrite(EnableL,40);

  digitalWrite(HighR, HIGH);
  digitalWrite(LowR, LOW);
  analogWrite(EnableR,90); 
} 

void Left3()
{
  digitalWrite(HighL, HIGH);
  digitalWrite(LowL, LOW);
  analogWrite(EnableL,30);

  digitalWrite(HighR, HIGH);
  digitalWrite(LowR, LOW);
  analogWrite(EnableR,90); 
}

void Right1()
{
  digitalWrite(HighL, HIGH);
  digitalWrite(LowL, LOW);
  analogWrite(EnableL,90);

  digitalWrite(HighR, HIGH);
  digitalWrite(LowR, LOW);
  analogWrite(EnableR,50); 
} 

void Right2()
{
  digitalWrite(HighL, HIGH);
  digitalWrite(LowL, LOW);
  analogWrite(EnableL,90);

  digitalWrite(HighR, HIGH);
  digitalWrite(LowR, LOW);
  analogWrite(EnableR,40); 
} 

void Right3()
{
  digitalWrite(HighL, HIGH);
  digitalWrite(LowL, LOW);
  analogWrite(EnableL,90);

  digitalWrite(HighR, HIGH);
  digitalWrite(LowR, LOW);
  analogWrite(EnableR,30); 
}

void Object()
{
  analogWrite(EnableL,0);
  analogWrite(EnableR,0);      //STOP
  delay(1000); 
//-----------------------------------
  digitalWrite(HighL, HIGH);
  digitalWrite(LowL, LOW);
  analogWrite(EnableL,0);
                              //LEFT
  digitalWrite(HighR, HIGH);
  digitalWrite(LowR, LOW);
  analogWrite(EnableR,90);
  delay(1500);  
//------------------------------------
  analogWrite(EnableL,0);
  analogWrite(EnableR,0);      //STOP
  delay(1000); 
//------------------------------------
  digitalWrite(HighL, HIGH);
  digitalWrite(LowL, LOW);
  analogWrite(EnableL,80);
  digitalWrite(HighR, HIGH);  //FORWARD
  digitalWrite(LowR, LOW);
  analogWrite(EnableR,80); 
  delay(300);
//------------------------------------
  analogWrite(EnableL,0);
  analogWrite(EnableR,0);      //STOP
  delay(1000); 
//-----------------------------------
  digitalWrite(HighL, HIGH);
  digitalWrite(LowL, LOW);
  analogWrite(EnableL,90);
                               //RIGHT
  digitalWrite(HighR, HIGH);
  digitalWrite(LowR, LOW);
  analogWrite(EnableR,0); 
  delay(2700);  
//------------------------------------
  analogWrite(EnableL,0);
  analogWrite(EnableR,0);      //STOP
  delay(1000);
//------------------------------------
  digitalWrite(HighL, HIGH);
  digitalWrite(LowL, LOW);
  analogWrite(EnableL,80);
  digitalWrite(HighR, HIGH);  //FORWARD
  digitalWrite(LowR, LOW);
  analogWrite(EnableR,80); 
  delay(500); 
}

void loop() 
{
  Data();
  if(data==0)
   {
     Forward();
   }
  else if(data==1)
   {
     Right1();
   }    
  else if(data==2)
   {
     Right2();
   }    
  else if(data==3)
   {
     Right3();
   }   
  else if(data==4)
   {
     Left1();
   } 
  else if(data==5)
   {
     Left2();
   }  
  else if(data==6)
   {
     Left3();
   }
  else if (data==7)
   {
     analogWrite(EnableL, 0);
     analogWrite(EnableR, 0); 
     delay(4000);  

     analogWrite(EnableL, 80);
     analogWrite(EnableR, 80); 
     delay(1000); 
   } 
  else if (data == 8)
   {
     Stop();
   } 
  else if (data == 9)
   {
     analogWrite(EnableL, 0);
     analogWrite(EnableR, 0); 
     delay(4000);  
   } 
  else if (data>9)
   {
     Stop();
   }
}
