// Event listener for file upload
document.getElementById('uploadForm').addEventListener('submit', async function (event) {
    event.preventDefault();

    const formData = new FormData();
    const dicomFile = document.getElementById('dicomFile').files[0];
    
    if (!dicomFile) {
        alert('Please select a file to upload');
        return;
    }
    
    formData.append('dicom_file', dicomFile);

    // Show the loading element
    document.getElementById('loading').style.display = 'block';

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }

        const data = await response.json();

        // Hide the loading element once done
        document.getElementById('loading').style.display = 'none';

        // Render each Plotly figure
        if (data.slices && data.slices.length > 0) {
            const container = document.getElementById('plot2DContainer');
            container.innerHTML = '';
            
            data.slices.forEach(slice => {
                // Create a container for each subplot
                const subplotDiv = document.createElement('div');
                subplotDiv.className = 'subplot';
                subplotDiv.id = `plot-${slice.view}`;
                
                // Add a caption
                const caption = document.createElement('p');
                caption.innerText = slice.view;
                subplotDiv.appendChild(caption);
                
                container.appendChild(subplotDiv);
                
                try {
                    // Parse the Plotly JSON data
                    const plotlyData = JSON.parse(slice.plotly_json);
                    
                    // Create the Plotly plot
                    Plotly.newPlot(
                        `plot-${slice.view}`, 
                        plotlyData.data, 
                        plotlyData.layout, 
                        {
                            responsive: true,
                            displayModeBar: false,
                        }
                    ).then(() => {
                        console.log(`Plot for ${slice.view} rendered successfully`);
                    }).catch(err => {
                        console.error(`Error rendering plot for ${slice.view}:`, err);
                    });
                    
                    // Add a download button
                    const downloadBtn = document.createElement('button');
                    downloadBtn.className = 'download-btn';
                    downloadBtn.innerHTML = '&#x2B07;'; // a simple down arrow icon
                    downloadBtn.onclick = function() {
                        Plotly.downloadImage(`plot-${slice.view}`, {
                            format: 'png',
                            filename: `${slice.view}`,
                        });
                    };
                    
                    subplotDiv.appendChild(downloadBtn);
                } catch (parseError) {
                    console.error('Error parsing Plotly JSON:', parseError);
                    subplotDiv.innerHTML += '<p class="error">Error rendering plot. Check console for details.</p>';
                }
            });
        } else {
            document.getElementById('plot2DContainer').innerHTML = '<p>No slice data received from the server.</p>';
        }
    } catch (error) {
        // Hide the loading element if an error occurs
        document.getElementById('loading').style.display = 'none';
        alert('Error processing the file: ' + error.message);
        console.error('Upload error:', error);
    }
});

// Disclosure modal logic
document.addEventListener('DOMContentLoaded', function () {
    if (!localStorage.getItem('disclosureAccepted')) {
      const modal = document.getElementById('disclosureModal');
      modal.style.display = 'block';
  
      document.getElementById('closeModal').addEventListener('click', function () {
        modal.style.display = 'none';
        localStorage.setItem('disclosureAccepted', 'true');
      });
  
      document.getElementById('modalOk').addEventListener('click', function () {
        modal.style.display = 'none';
        localStorage.setItem('disclosureAccepted', 'true');
      });
    }
    
    // Add console message to help with debugging
    console.log('DOM fully loaded. Ready for file uploads.');
});

const uploadButton = document.querySelector('#uploadForm button');
const plotSection = document.querySelector('#plot2D-section');

// Event listener for when the upload button is clicked
uploadButton.addEventListener('click', function() {
  // Optionally, you can add a check here to ensure the file is actually uploaded before showing the section
  plotSection.style.display = 'block';  // Show the plot section
});


// Get the button by its ID
const guideButton = document.querySelector('#guideButton');

// URL of the PDF file
const pdfUrl = 'CMR_Planning_Guide.pdf';

// Event listener for the button click
guideButton.addEventListener('click', function() {
  window.open(pdfUrl, '_blank'); // Open the PDF in a new tab
});
