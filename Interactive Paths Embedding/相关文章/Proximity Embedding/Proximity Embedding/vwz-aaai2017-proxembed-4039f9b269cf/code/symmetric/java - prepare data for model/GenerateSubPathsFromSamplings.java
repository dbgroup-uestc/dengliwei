package dataPrepare.ProxEmbed;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Generate sub-paths by samplings.
 */
public class GenerateSubPathsFromSamplings {

	static String nodes_path=Config.NODES_PATH;		//定义节点路径
	static String conditional_random_walk_sampling_paths=Config.SAVE_PATH_FOR_RANDOMWALK_SAMPLINGS;	//定义random walk采样的文件的路径（这个采样是在randomWalkSampling中进行的）
	static String truncated_type_name=Config.TRUNCATED_TYPE_NAME;	//定义截断的类型
	static String subpaths_save_path=Config.SUBPATHS_SAVE_PATH;		//定义subpath的存放路径
	static int longest_length_for_window=Config.LONGEST_ANALYSE_LENGTH_FOR_SAMPLING;	//如果subpath的长度（去环之前）大于maxWindowLen，那么退出循环的长度（去环之前）大于maxWindowLen，那么退出循环
	static int longest_lenght_for_subpaths=Config.LONGEST_LENGTH_FOR_SUBPATHS;	//最长的子采样长度
	
	public static void main(String[] args) {
		GenerateSubPathsFromSamplings g=new GenerateSubPathsFromSamplings();
		g.generateSubPathsFromSamplings(
				nodes_path, 
				conditional_random_walk_sampling_paths, 
				truncated_type_name, 
				subpaths_save_path, 
				longest_length_for_window,
				longest_lenght_for_subpaths);
	}

	/**
	 * Generate sub-paths by samplings.
	 */
	public void generateSubPathsFromSamplings(String nodesPath,String samplingsPath,String truncatedNodeType,String subPathsSavePath,int window_maxlen,int subpath_maxlen){
		Set<Integer> truncatedNodeIds=new HashSet<Integer>();
		Set<String> truncatedTypes=new HashSet<String>();
		String[] arr=truncatedNodeType.split(" ");	//可能有多个截断类型，将其均放入arr字符串中
		truncatedTypes.addAll(Arrays.asList(arr));	//将所有的截断类型添加到集合truncatedTypes中
		BufferedReader br=null;
		arr=null;
		try {	//设置truncatedNodeIds
			br = new BufferedReader(new InputStreamReader(new FileInputStream(nodesPath), "UTF-8"));
			String temp = null;
			while ((temp = br.readLine()) != null ) {
				temp=temp.trim();
				if(temp.length()>0){
					arr=temp.split("	");	//读取node文件，以tab作为分割，得到一个长度为3的数组
					if(truncatedTypes.contains(arr[1])){	//如果该节点的类型被包含在所给出的截断类型中，就将该节点的id添加到truncatedNodeIds中
						truncatedNodeIds.add(Integer.parseInt(arr[0]));
					}
				}
			}
		} catch (Exception e2) {
			e2.printStackTrace();
		}
		finally{
			try {
				if(br!=null){
					br.close();
					br=null;
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		FileWriter writer =null;
		String t=null;
		List<Integer> path=new ArrayList<Integer>();
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(samplingsPath), "UTF-8"));
			writer = new FileWriter(subPathsSavePath);
			String temp = null;
			while ((temp = br.readLine()) != null ) {
				temp=temp.trim();
				if(temp.length()>0){
					path.clear();
					arr=temp.split(" ");
					for(String s:arr){
						path.add(Integer.parseInt(s));	//path是节点的id的列表
					}
					t=analyseOnePath(path, truncatedNodeIds, window_maxlen, subpath_maxlen);	//根据之前所生成的path，获取子路径
					if(t.length()>0){	//如果所采样的子路径不为空，则将其写入子路径存放的文件
						writer.write(t);
						writer.flush();
					}
				}
			}
		} catch (Exception e2) {
			e2.printStackTrace();
		}
		finally{
			try {
				if(writer!=null){
					writer.close();
					writer=null;
				}
				if(br!=null){
					br.close();
					br=null;
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	/**
	 * Generate sub-paths by one specific sampling path.
	 */
	private String analyseOnePath(List<Integer> path,Set<Integer> truncatedNodeIds,int maxWindowLen,int maxSubpathLen){
		StringBuilder sb=new StringBuilder();
		int start=0;
		int end=0;
		List<Integer> subpath=new ArrayList<Integer>();
		for(int i=0;i<path.size();i++){
			start=path.get(i);	//节点从0开始，进行子采样
			if(!truncatedNodeIds.contains(start)){	//如果start节点不是可以截断的节点，则跳过本次循环，将start在path上向后移动一个节点，直到path上的所有节点都被检测完成
				continue;
			}
			for(int j=i+1;j<path.size();j++){
				end=path.get(j);	//从开始节点的下一个开始，寻找可以截断的节点
				if(!truncatedNodeIds.contains(end)){	//如果end不是可以截断的节点，则跳过此次循环
					continue;
				}
				
				if(maxWindowLen>0 && (j-i)>maxWindowLen){	//如果子路径的长度（去环之前）大于maxWindowLen，那么退出循环
					break;
				}
				
				subpath.clear();	//每次循环都清空子路径
				for(int x=i;x<=j;x++){
					subpath.add(path.get(x)+0);	//将path[i]到path[j]的路径定义为一条子路径，这个+0是什么意思？没搞明白。
				}
				List<Integer> subpathNoRepeat=deleteRepeat(subpath);	//删除path中存在的环
				if(subpathNoRepeat.size()<2){	//如果删除环之后子路径的长度小于2，那么说明该子路径的首尾是相同的
					subpathNoRepeat=null;		//此时直接跳过本次循环即可，采样下一个子路径
					continue;
				}
				
				if(maxSubpathLen>0 && subpathNoRepeat.size()>maxSubpathLen){//如果子路径的长度（去环之后）大于maxSubpathLen，那么退出循环
					continue;
				}
				
				sb.append(path.get(i)+"	"+path.get(j)+"	");		//将子路径的首尾节点的id添加到sb字符串中
				for(int x=0;x<subpathNoRepeat.size();x++){		//将子路径所包含的所有节点，按照顺序添加到sb字符串中
					sb.append(subpathNoRepeat.get(x)+" ");
				}
				sb.append("\r\n");	//添加换行符号
				subpathNoRepeat=null;
			}
		}
		return sb.toString();	//将所生成的子路径返回，子路径类似于:1 7 1 2 3 7 \r\n2 4 2 5 4\n\r
	}							//其中前两个字符分别表示子路径的首尾节点，其后是子路径具体所包含的节点
	
	/**
	 * Delete repeat segments for sub-paths
	 */
	public List<Integer> deleteRepeat(List<Integer> path){	//	参数path是节点id的列表
		Map<Integer,Integer> map=new HashMap<Integer,Integer>();
		int node=0;
		List<Integer> result=new ArrayList<Integer>();
		int formerIndex=0;
		for(int i=0;i<path.size();i++){
			node=path.get(i);
			if(!map.containsKey(node)){		//将节点的id与它在子路径上的位置形成一个map
				map.put(node, i);			//如果map中不存在该节点，那么将该节点与其在子路径上的位置添加到map中
			}
			else{	//如果该map中存在该节点
				formerIndex=map.get(node);	//获取该节点在map中的序号
				for(int j=formerIndex;j<i;j++){		//从该节点开始，一直到最后，移除map中所存在的映射
					map.remove(path.get(j));
					path.set(j, -1);	//将所删除的节点的值设置为-1
				}
				map.put(node, i);
			}
		}
		for(int i=0;i<path.size();i++){
			if(path.get(i)!=-1){
				result.add(path.get(i));	//将path中值不是-1的节点的id存入result中
			}
		}
		return result;
	}
}
