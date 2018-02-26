import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class Screen {

	public int width;
	public int height;
	public int[] pixels;
	public float[][] canv;
	public int prevMouseB;
	public boolean record;
	
	public float[][] weight0;
	public float[][] weight1;
	public float[][] weight2;

	public Screen(int width, int height) {
		this.prevMouseB = Mouse.getButton();
		this.width = width;
		this.height = height;
		canv = new float[1][28 * 28];
		pixels = new int[width * height];
		weight0 = new float[28 * 28][16];
		weight1 = new float[16][16];
		weight2 = new float[16][10];
		
		try{
			InputStream in = getClass().getResourceAsStream("/w0.txt");
			//FileReader f = new FileReader("res/w0.txt");
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String s = "";
			int counter = 0;
			while((s = br.readLine()) != null){
				extractData(s, 0, counter);
				counter++;
			}
			in = getClass().getResourceAsStream("/w1.txt");
			//f = new FileReader("res/w1.txt");
			br = new BufferedReader(new InputStreamReader(in));
			s = "";
			counter = 0;
			while((s = br.readLine()) != null){
				extractData(s, 1, counter);
				counter++;
			}
			in = getClass().getResourceAsStream("/w2.txt");
			//f = new FileReader("res/w2.txt");
			br = new BufferedReader(new InputStreamReader(in));
			s = "";
			counter = 0;
			while((s = br.readLine()) != null){
				extractData(s, 2, counter);
				counter++;
			}
			br.close();
		}catch(FileNotFoundException ex){
			System.out.println("File move");
		} catch (IOException ex) {
			System.out.println("Reading problem");
		}
	}
	
	public void extractData(String s, int inpNo, int index){
		String w = "";
		int count = 0;
		for(int i = 0; i < s.length(); i++){
			char c = s.charAt(i);
			if(c != ','){
				w += c;
			}else{
				if(inpNo == 0){
					weight0[index][count] = Float.parseFloat(w);
				}else if(inpNo == 1){
					weight1[index][count] = Float.parseFloat(w);
				}else if(inpNo == 2){
					weight2[index][count] = Float.parseFloat(w);
				}
				count++;
				w = "";
			}
		}
		if(inpNo == 0){
			weight0[index][count] = Float.parseFloat(w);
		}else if(inpNo == 1){
			weight1[index][count] = Float.parseFloat(w);
		}else if(inpNo == 2){
			weight2[index][count] = Float.parseFloat(w);
		}
	}

	public void render() {
		for(int i = 0; i < this.width; i++){
			for(int j = 0; j < this.height; j++){
				int col = (canv[0][(28 * i / this.width) + (28 * j / this.height) * 28] == 0) ? 0xffffff : 0x000000;
				this.pixels[i + j * width] = col;
			}
		}
	}
	
	public void update() {
		
		if(mapX(Mouse.getX()) + mapY(Mouse.getY()) * 28 >= 0 && mapX(Mouse.getX()) + mapY(Mouse.getY()) * 28 < 28 * 28){
			if(this.record){
				for(int k = -1; k < 2; k++){
					for(int l = -1; l < 2; l++){
						if((mapX(Mouse.getX()) + k) + (mapY(Mouse.getY()) + l) * 28 >= 0 && (mapX(Mouse.getX()) + k) + (mapY(Mouse.getY()) + l) * 28 < 28 * 28){
							canv[0][(mapX(Mouse.getX()) + k) + (mapY(Mouse.getY()) + l) * 28] = 0.5f;
						}
					}
				}
				canv[0][mapX(Mouse.getX()) + mapY(Mouse.getY()) * 28] = 1;
			}else{
				for (int i = 0; i < canv[0].length; i++) {
					canv[0][i] = 0;
				}
			}
		}
	}
	
	public float[][] mult(float[][] m1, float[][] m2){
		float[][] result = new float[m1.length][m2[0].length];
		for(int i = 0; i < m1.length; i++){
			for(int j = 0; j < m2[0].length; j++){
				for(int k = 0; k < m1[0].length; k++){
					result[i][j] += m1[i][k] * m2[k][j];
				}
			}
		}
		return result;
	}

	public float[][] lin(float[][] x){
		float[][] result = new float[x.length][x[0].length];
		for(int i = 0; i < x.length; i++){
			for(int j = 0; j < x[i].length; j++){
				result[i][j] = 1 / (1 + (float)Math.exp(-1 * x[i][j]));
			}
		}
		return result;
	}
	
	public int predict(){
		//System.out.println(Math.E);
		//System.out.println(Math.exp(Math.E));
		// canv 784 // w0 784 16 w1 16 16 w2 16 10
		float[][] l1 = lin(mult(canv, weight0));
		float[][] l2 = lin(mult(l1, weight1));
		float[][] l3 = lin(mult(l2, weight2));
		float max = 0;
		int index = -1;
		for(int i = 0; i < 10; i++){
			if(l3[0][i] > max){
				max = l3[0][i];
				index = i;
			}
		}
		return index;
	}
	
	public int mapX(int var){
		return 28 * var / (this.width * 3);
	}
	
	public int mapY(int var){
		return 28 * var/ (this.height * 3);
	}

	public void clear() {
		for (int i = 0; i < pixels.length; i++) {
			pixels[i] = 0;
		}
	}
}
