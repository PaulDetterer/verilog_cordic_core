/*  file:        derotator.sv
    author:      Paul Detterer
    release:     17/11/2020

    First Quadrant Cordic based Derotator 

    Intended to be used for frequency down conversion in digital baseband

    The design is inspired by the CORDIC implementation of Dale Drinkard from OpenCores

*/

module derotator 
#(
  parameter integer iteration = 0,
  parameter BW=12,
  parameter ABW=12,
  parameter signed [ABW-1:0] tangle = 0
)
(
  input wire clk,
  input wire rst,
  input wire signed  [BW-1:0]    x_i,
  input wire signed  [BW-1:0]    y_i,
  input wire signed  [ABW-1:0] z_i,
  output wire signed [BW-1:0]    x_o,
  output wire signed [BW-1:0]    y_o,
  output wire signed [ABW-1:0] z_o
  );
  
  reg signed [BW-1:0] x_1;
  reg signed [BW-1:0] y_1;
  reg signed [ABW-1:0] z_1;
  wire signed [BW-1:0] x_i_shifted = x_i >>> iteration;
  wire signed [BW-1:0] y_i_shifted = y_i >>> iteration;

  always_comb
      if (z_i < 0) begin // Rotate Mode
        x_1 <= x_i - y_i_shifted; //shifter(y_1,i); //(y_1 >> i);
        y_1 <= y_i + x_i_shifted; //shifter(x_1,i); //(x_1 >> i);
        z_1 <= z_i + tangle;
      end else begin
        x_1 <= x_i + y_i_shifted; //shifter(y_1,i); //(y_1 >> i);
        y_1 <= y_i - x_i_shifted; //shifter(x_1,i); //(x_1 >> i);
        z_1 <= z_i - tangle;
      end
  assign x_o = x_1;
  assign y_o = y_1;
  assign z_o = z_1;
endmodule
