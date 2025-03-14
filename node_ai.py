import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from google import genai
import json
import os
import time
import uuid
from PIL import Image
import io

# Configure page
st.set_page_config(
    page_title="NODE.AI - Decision Tree Story Writer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-bottom: 1rem;
    }
    .story-box {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 20px;
    }
    .path-option {
        background-color: #EFF6FF;
        padding: 15px;
        border-radius: 8px;
        border-left: 3px solid #60A5FA;
        margin-bottom: 10px;
        cursor: pointer;
    }
    .path-option:hover {
        background-color: #DBEAFE;
    }
    .selected-path {
        background-color: #DBEAFE;
        border-left: 3px solid #2563EB;
    }
    .story-title {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Gemini API client
def initialize_gemini_client(api_key):
    return genai.Client(api_key=api_key)

# Story Node class
class StoryNode:
    def __init__(self, content, node_id=None, parent_id=None, weight=1.0):
        self.id = node_id if node_id else str(uuid.uuid4())
        self.content = content
        self.parent_id = parent_id
        self.children = []
        self.weight = weight
    
    def add_child(self, child_node):
        self.children.append(child_node)
        return child_node.id
    
    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "parent_id": self.parent_id,
            "weight": self.weight,
            "children": [child.id for child in self.children]
        }

# Story Tree class
class StoryTree:
    def __init__(self):
        self.nodes = {}
        self.root_id = None
        self.graph = nx.DiGraph()
    
    def add_root(self, content):
        node = StoryNode(content)
        self.nodes[node.id] = node
        self.root_id = node.id
        self.graph.add_node(node.id, content=content[:50] + "..." if len(content) > 50 else content)
        return node.id
    
    def add_node(self, content, parent_id, weight=1.0):
        if parent_id not in self.nodes:
            return None
        
        node = StoryNode(content, parent_id=parent_id, weight=weight)
        self.nodes[node.id] = node
        self.nodes[parent_id].add_child(node)
        
        self.graph.add_node(node.id, content=content[:50] + "..." if len(content) > 50 else content)
        self.graph.add_edge(parent_id, node.id, weight=weight)
        
        return node.id
    
    def get_node(self, node_id):
        return self.nodes.get(node_id)
    
    def get_path_to_node(self, node_id):
        path = []
        current_id = node_id
        
        while current_id is not None:
            node = self.nodes.get(current_id)
            if node:
                path.append(node)
                current_id = node.parent_id
            else:
                break
        
        return list(reversed(path))
    
    def visualize(self):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Node sizes based on text length
        node_sizes = [min(len(self.nodes[n].content), 1000) + 500 for n in self.graph.nodes()]
        
        # Edge widths based on weights
        edge_weights = nx.get_edge_attributes(self.graph, 'weight')
        edge_widths = [w*2 for w in edge_weights.values()]
        
        # Draw nodes with different colors for leaf nodes
        leaf_nodes = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        non_leaf_nodes = [n for n in self.graph.nodes() if n not in leaf_nodes]
        
        # Draw non-leaf nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                              nodelist=non_leaf_nodes,
                              node_size=[node_sizes[i] for i, n in enumerate(self.graph.nodes()) if n in non_leaf_nodes],
                              node_color='skyblue',
                              alpha=0.8)
        
        # Draw leaf nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                              nodelist=leaf_nodes,
                              node_size=[node_sizes[i] for i, n in enumerate(self.graph.nodes()) if n in leaf_nodes],
                              node_color='lightgreen',
                              alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, width=edge_widths, alpha=0.7, edge_color='gray', arrows=True, arrowsize=15)
        
        # Draw labels
        labels = {node: f"{node_id[:4]}..." for node_id, node in enumerate(self.graph.nodes())}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=10, font_weight='bold')
        
        plt.axis('off')
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        return img
    
    def save(self, filename):
        data = {
            "root_id": self.root_id,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()}
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.root_id = data["root_id"]
        self.nodes = {}
        self.graph = nx.DiGraph()
        
        # First pass: create all nodes
        for nid, node_data in data["nodes"].items():
            node = StoryNode(
                node_data["content"], 
                node_id=nid, 
                parent_id=node_data.get("parent_id"),
                weight=node_data.get("weight", 1.0)
            )
            self.nodes[nid] = node
            self.graph.add_node(nid, content=node_data["content"][:50] + "..." if len(node_data["content"]) > 50 else node_data["content"])
        
        # Second pass: connect nodes
        for nid, node_data in data["nodes"].items():
            parent_id = node_data.get("parent_id")
            if parent_id is not None:
                self.graph.add_edge(parent_id, nid, weight=node_data.get("weight", 1.0))
            
            for child_id in node_data.get("children", []):
                if child_id in self.nodes and child_id not in [child.id for child in self.nodes[nid].children]:
                    self.nodes[nid].children.append(self.nodes[child_id])

# Story Generator class
class StoryGenerator:
    def __init__(self, gemini_client):
        self.client = gemini_client
        self.story_tree = StoryTree()
        self.current_node_id = None
        self.model = "gemini-2.0-flash"
    
    def generate_initial_situation(self, genre, protagonist, antagonist, setting):
        prompt = f"""
        Generate an engaging opening situation for a {genre} story featuring:
        - Protagonist: {protagonist}
        - Antagonist: {antagonist}
        - Setting: {setting}
        
        The situation should be compelling and open-ended, allowing for multiple possible developments.
        Write only the opening paragraph (100-150 words) that sets the scene and introduces the conflict.
        Make it vivid and immersive.
        """
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        
        situation = response.text.strip()
        self.current_node_id = self.story_tree.add_root(situation)
        return situation
    
    def generate_possible_paths(self, current_situation, num_paths=3):
        prompt = f"""
        Based on the following situation in a story:
        
        "{current_situation}"
        
        Generate {num_paths} distinct, compelling continuations that could follow. Each continuation 
        should be a different possible path the story could take, representing different narrative possibilities.
        
        For each path:
        1. Write a single paragraph (80-120 words) that continues the story in an interesting direction
        2. Make each path clearly different from the others
        3. Ensure each path maintains narrative coherence with what came before
        4. Include some action, dialogue, or character development
        
        Format each path as:
        Path 1: [continuation text]
        Path 2: [continuation text]
        Path 3: [continuation text]
        """
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        
        # Parse the response to extract the different paths
        raw_text = response.text
        paths = []
        
        for i in range(1, num_paths + 1):
            path_marker = f"Path {i}:"
            next_marker = f"Path {i+1}:" if i < num_paths else None
            
            start_idx = raw_text.find(path_marker)
            if start_idx == -1:
                continue
                
            start_idx += len(path_marker)
            end_idx = raw_text.find(next_marker, start_idx) if next_marker else len(raw_text)
            
            if end_idx > start_idx:
                path_text = raw_text[start_idx:end_idx].strip()
                paths.append(path_text)
        
        return paths
    
    def add_paths(self, paths):
        if not self.current_node_id:
            return []
        
        # Add each path as a child node with weight
        path_ids = []
        for i, path in enumerate(paths):
            # Assign decreasing weights to paths (first path has highest weight)
            weight = 1.0 - (i * 0.2)
            if weight < 0.2:
                weight = 0.2
            
            path_id = self.story_tree.add_node(path, self.current_node_id, weight)
            path_ids.append(path_id)
        
        return path_ids
    
    def select_path(self, path_id):
        """Select a specific path and update current node"""
        node = self.story_tree.get_node(path_id)
        if node:
            self.current_node_id = path_id
            return node.content
        return None
    
    def get_full_story(self):
        """Retrieves the full story path from root to current node"""
        if self.current_node_id is None:
            return ""
            
        path = self.story_tree.get_path_to_node(self.current_node_id)
        story_sections = [node.content for node in path]
        return "\n\n".join(story_sections)
    
    def generate_custom_continuation(self, current_story, user_input):
        """Generate a continuation based on user input"""
        prompt = f"""
        Current story so far:
        
        {current_story}
        
        User wants the story to continue with this idea:
        {user_input}
        
        Write a single paragraph (100-150 words) that continues the story based on the user's idea.
        Make it coherent with what came before, but incorporate the user's direction.
        Include vivid details, and either dialogue, action, or character development.
        """
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        
        continuation = response.text.strip()
        new_node_id = self.story_tree.add_node(continuation, self.current_node_id, 1.0)
        self.current_node_id = new_node_id
        return continuation
    
    def generate_ending(self, current_story):
        """Generate a satisfying ending for the story"""
        prompt = f"""
        Here is a story that needs a satisfying conclusion:
        
        {current_story}
        
        Write a final paragraph (100-150 words) that provides a satisfying ending to this story.
        The ending should:
        1. Resolve the main conflict or tension
        2. Provide emotional closure for the main characters
        3. Leave the reader with a sense of completion
        4. Match the tone and style of the existing story
        """
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        
        ending = response.text.strip()
        end_node_id = self.story_tree.add_node(ending, self.current_node_id, 1.0)
        self.current_node_id = end_node_id
        return ending
    
    def get_story_title(self, full_story):
        """Generate a title for the story"""
        prompt = f"""
        Based on the following story, generate a compelling, creative title (5-10 words):
        
        {full_story}
        
        The title should:
        1. Capture the essence of the story
        2. Be intriguing and memorable
        3. Reflect the genre and tone
        
        Return ONLY the title, nothing else.
        """
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        
        title = response.text.strip()
        return title
    
    def save_story(self, filename):
        """Saves the current story tree state"""
        self.story_tree.save(filename)
    
    def load_story(self, filename):
        """Loads a story tree from file"""
        self.story_tree.load(filename)
        # Set current node to the last node in the deepest path
        if self.story_tree.root_id is not None:
            self.current_node_id = self.story_tree.root_id

# Main Streamlit app
def main():
    # Initialize session states
    if 'api_key_set' not in st.session_state:
        st.session_state.api_key_set = False
    
    if 'story_generator' not in st.session_state:
        st.session_state.story_generator = None
    
    if 'current_paths' not in st.session_state:
        st.session_state.current_paths = []
    
    if 'path_ids' not in st.session_state:
        st.session_state.path_ids = []
    
    if 'story_title' not in st.session_state:
        st.session_state.story_title = "Untitled Story"
    
    if 'story_completed' not in st.session_state:
        st.session_state.story_completed = False
    
    # App header
    st.markdown("<h1 class='main-header'>NODE.AI: Decision Tree Story Writer</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='text-align: center; font-size: 1.2rem; margin-bottom: 30px;'>
        Create branching narratives with AI assistance. Choose your path or forge your own.
        </p>
        """, 
        unsafe_allow_html=True
    )
    
    # Sidebar for API key and configuration
    with st.sidebar:
        st.header("📝 Configuration")
        api_key = st.text_input("Enter Gemini API Key", type="password")
        
        if st.button("Set API Key") and api_key:
            try:
                client = initialize_gemini_client(api_key)
                st.session_state.story_generator = StoryGenerator(client)
                st.session_state.api_key_set = True
                st.success("API key set successfully!")
            except Exception as e:
                st.error(f"Error setting API key: {str(e)}")
        
        if st.session_state.api_key_set:
            st.success("✅ API key is set")
            
            # Story operations
            st.subheader("📚 Story Operations")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Story"):
                    if st.session_state.story_generator:
                        try:
                            filename = f"story_{int(time.time())}.json"
                            st.session_state.story_generator.save_story(filename)
                            st.success(f"Story saved as {filename}")
                        except Exception as e:
                            st.error(f"Error saving story: {str(e)}")
            
            with col2:
                saved_stories = [f for f in os.listdir() if f.startswith("story_") and f.endswith(".json")]
                if saved_stories:
                    selected_story = st.selectbox("Load Story", saved_stories)
                    if st.button("Load"):
                        try:
                            st.session_state.story_generator.load_story(selected_story)
                            st.success("Story loaded successfully!")
                            st.session_state.current_paths = []
                            st.session_state.path_ids = []
                            st.session_state.story_completed = False
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error loading story: {str(e)}")
        
        # About section
        st.markdown("---")
        st.markdown(
            """
            ### About NODE.AI
            
            NODE.AI uses decision tree-based storytelling with LLM assistance to create branching narratives.
            
            - 🌳 Create branching story paths
            - 🔄 Choose from AI suggestions or write your own
            - 📊 Visualize your story structure
            - 💾 Save and load stories
            
            Built with Streamlit and Gemini API
            """
        )
    
    # Main content area
    if not st.session_state.api_key_set:
        st.warning("Please set your Gemini API key in the sidebar to begin.")
        
        # Display sample story tree visualization
        st.markdown("### Sample Story Tree Visualization")
        sample_img = Image.open("arch.png") if os.path.exists("arch.png") else None
        if sample_img:
            st.image(sample_img, caption="Sample story tree visualization")
        
    else:
        # Story creation form
        if not hasattr(st.session_state.story_generator, 'current_node_id') or st.session_state.story_generator.current_node_id is None:
            st.markdown("<h2 class='sub-header'>Create a New Story</h2>", unsafe_allow_html=True)
            
            with st.form("story_parameters"):
                col1, col2 = st.columns(2)
                
                with col1:
                    genre = st.selectbox(
                        "Genre", 
                        ["Fantasy", "Sci-Fi", "Mystery", "Romance", "Horror", "Adventure", 
                         "Historical Fiction", "Comedy", "Thriller", "Drama"]
                    )
                    protagonist = st.text_input("Protagonist Name", "Alex")
                
                with col2:
                    antagonist = st.text_input("Antagonist/Challenge", "Morgan")
                    setting = st.text_input("Setting", "A small coastal town with a mysterious lighthouse")
                
                submit = st.form_submit_button("Begin Your Story")
                
                if submit:
                    with st.spinner("Creating your story world..."):
                        initial_situation = st.session_state.story_generator.generate_initial_situation(
                            genre, protagonist, antagonist, setting
                        )
                        st.experimental_rerun()
        
        # Display current story and options
        else:
            # Get the full story so far
            full_story = st.session_state.story_generator.get_full_story()
            
            # Generate title if not already done
            if st.session_state.story_title == "Untitled Story" and len(full_story) > 100:
                with st.spinner("Generating title..."):
                    st.session_state.story_title = st.session_state.story_generator.get_story_title(full_story)
            
            # Display the story title and content
            st.markdown(f"<h1 class='story-title'>{st.session_state.story_title}</h1>", unsafe_allow_html=True)
            
            # Story content
            st.markdown("<div class='story-box'>", unsafe_allow_html=True)
            paragraphs = full_story.split('\n\n')
            for p in paragraphs:
                st.markdown(f"<p>{p}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Story is not completed yet
            if not st.session_state.story_completed:
                # Generate and display path options if we don't already have them
                if not st.session_state.current_paths:
                    current_node = st.session_state.story_generator.story_tree.get_node(
                        st.session_state.story_generator.current_node_id
                    )
                    
                    # If the current node has no children, generate new paths
                    if not current_node.children:
                        with st.spinner("Generating possible story paths..."):
                            paths = st.session_state.story_generator.generate_possible_paths(current_node.content)
                            path_ids = st.session_state.story_generator.add_paths(paths)
                            
                            st.session_state.current_paths = paths
                            st.session_state.path_ids = path_ids
                
                # Display path options
                if st.session_state.current_paths:
                    st.markdown("<h2 class='sub-header'>Choose Your Path</h2>", unsafe_allow_html=True)
                    
                    # Display AI-generated paths
                    for i, (path, path_id) in enumerate(zip(st.session_state.current_paths, st.session_state.path_ids)):
                        path_node = st.session_state.story_generator.story_tree.get_node(path_id)
                        
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.markdown(f"<div class='path-option'>", unsafe_allow_html=True)
                            st.markdown(f"<strong>Path {i+1}:</strong> {path}", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        with col2:
                            if st.button(f"Choose", key=f"path_{i}"):
                                st.session_state.story_generator.select_path(path_id)
                                st.session_state.current_paths = []
                                st.session_state.path_ids = []
                                st.experimental_rerun()
                    
                    # Custom path option
                    st.markdown("<h3>Or Write Your Own Continuation</h3>", unsafe_allow_html=True)
                    
                    with st.form("custom_path"):
                        user_continuation = st.text_area(
                            "Describe how you want the story to continue",
                            height=100,
                            placeholder="The protagonist decides to..."
                        )
                        submit_custom = st.form_submit_button("Continue with My Idea")
                        
                        if submit_custom and user_continuation:
                            with st.spinner("Crafting your continuation..."):
                                st.session_state.story_generator.generate_custom_continuation(
                                    full_story, user_continuation
                                )
                                st.session_state.current_paths = []
                                st.session_state.path_ids = []
                                st.experimental_rerun()
                    
                    # Option to end the story
                    if st.button("End the Story"):
                        with st.spinner("Creating a satisfying ending..."):
                            st.session_state.story_generator.generate_ending(full_story)
                            st.session_state.story_completed = True
                            st.session_state.current_paths = []
                            st.session_state.path_ids = []
                            st.experimental_rerun()
            
            # Story is completed
            else:
                st.success("Your story is complete! You can save it or start a new one.")
                
                if st.button("Start New Story"):
                    st.session_state.story_generator = StoryGenerator(
                        initialize_gemini_client(api_key)
                    )
                    st.session_state.current_paths = []
                    st.session_state.path_ids = []
                    st.session_state.story_title = "Untitled Story"
                    st.session_state.story_completed = False
                    st.experimental_rerun()
            
            # Visualize the story tree
            st.markdown("<h2 class='sub-header'>Story Structure Visualization</h2>", unsafe_allow_html=True)
            
            with st.spinner("Generating visualization..."):
                try:
                    fig = st.session_state.story_generator.story_tree.visualize()
                    st.image(fig, caption="Your story's decision tree", use_column_width=True)
                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")

if __name__ == "__main__":
    main()